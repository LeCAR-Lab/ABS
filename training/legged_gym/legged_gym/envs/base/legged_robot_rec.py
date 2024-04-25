# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
# from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, yaw_quat, circle_ray_query
# from legged_gym.utils.helpers import class_to_dict
# from .legged_robot_config import LeggedRobotCfg
# from .legged_robot_pos_config import LeggedRobotPosCfg

class LeggedRobotRec(LeggedRobot):
    # cfg : LeggedRobotRecCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(1.0, 1.0, (len(env_ids), self.num_dof), device=self.device)
        if self.cfg.domain_rand.randomize_init_dof:  # xy position randomized
            # add the bias position for standing
            stand_bias = torch.zeros_like(self.dof_pos[env_ids])
            stand_bias[:,0::3] += self.cfg.domain_rand.stand_bias3[0]
            stand_bias[:,1::3] += self.cfg.domain_rand.stand_bias3[1]
            stand_bias[:,2::3] += self.cfg.domain_rand.stand_bias3[2]
            self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(self.cfg.domain_rand.init_dof_factor[0], self.cfg.domain_rand.init_dof_factor[1],\
                                             (len(env_ids), self.num_dof), device=self.device) + stand_bias
        self.dof_vel[env_ids] = self.dof_vel[env_ids].uniform_(-8.,8.)  # dangerous init

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        if self.cfg.asset.load_dynamic_object: env_ids_int32 = env_ids_int32 * (1 + self.cfg.asset.object_num)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _update_terrain_curriculum(self, env_ids):
        if not self.init_done:
            return
        move_up = torch.square( (self.base_lin_vel[env_ids, 0] - self.commands[env_ids, 0]).clip(min=0.) ) + \
                            torch.square(self.base_lin_vel[env_ids, 1]-self.commands[env_ids, 1]) < self.cfg.rewards.walkback_sigma * 0.5  # walk backward
        move_down = self.episode_length_buf[env_ids] < self.max_episode_length * 0.9  # cannot survive
        move_up *= ~move_down
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact

        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.empty_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:4] = 0. # contact
        noise_vec[4:7] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[7:10] = noise_scales.gravity * noise_level
        noise_vec[10:13] = 0. # commands
        noise_vec[13:25] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[25:37] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[37:49] = 0. # previous actions
        if self.cfg.sensors.ray2d.enable:
            noise_vec[49:] = noise_scales.ray2d * noise_level * self.obs_scales.ray2d
        return noise_vec

    def compute_cost(self):
        termination = self._reward_termination()
        self.extras['cost'] = termination > 0

    # def compute_cost(self):
    #     ...

    def compute_observations(self):
        """ Computes observations
        """
        if self.cfg.asset.load_dynamic_object:
            for _obj in range(self.cfg.asset.object_num):
                obj_relpos = self.root_states_obj[_obj][:, 0:3] - self.root_states[:, 0:3]
                self.obj_relpos[_obj][:] = quat_rotate_inverse(yaw_quat(self.base_quat[:]), obj_relpos)
        if self.cfg.sensors.ray2d.enable and self.cfg.asset.load_dynamic_object:
            self.ray2d_obs[:] = 99999.9
            for _obj in range(self.cfg.asset.object_num):
                _obj_type = _obj % (len(self.object_asset_list))
                _radius_obj = self.object_size_list[_obj_type]
                this_ray2d_obs = circle_ray_query(self.ray2d_x0, self.ray2d_y0, self.ray2d_thetas, self.obj_relpos[_obj][:,:2], radius=_radius_obj, min_=self.ray2d_range[0], max_=self.ray2d_range[1])
                self.ray2d_obs = torch.minimum(self.ray2d_obs, this_ray2d_obs)

        self.obs_buf = torch.cat((  self.contact_filt.float() * 2 - 1.0, # 0:4
                                    self.base_ang_vel  * self.obs_scales.ang_vel, # 4:7
                                    self.projected_gravity, # 7:10
                                    self.commands,  # 10:13
                                    (self.dof_pos - self.default_dof_pos - self.dof_bias) * self.obs_scales.dof_pos, # 13:25
                                    self.dof_vel * self.obs_scales.dof_vel, # 25:37
                                    self.actions # 37:49
                                    ),dim=-1)  # append extero after this
        # add perceptive inputs if not blind
        if self.cfg.sensors.ray2d.enable:
            if self.add_noise and self.cfg.sensors.ray2d.illusion:
                safe_tgt_dist = torch.norm(self.commands[:, :2], dim=-1).unsqueeze(1) + 0.35
                hallu_ = self.ray2d_obs > safe_tgt_dist
                self.ray2d_obs += hallu_ * torch.zeros_like(self.ray2d_obs).uniform_(0,1) * (safe_tgt_dist - self.ray2d_obs)
            if self.cfg.sensors.ray2d.log2:
                ray2d_ = torch.log2(self.ray2d_obs) * self.obs_scales.ray2d
            else:
                ray2d_ = self.ray2d_obs * self.obs_scales.ray2d
            self.obs_buf = torch.cat((self.obs_buf, ray2d_), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    #### rewards
    def _reward_walkback(self):
        lin_vel_error = torch.square( (self.base_lin_vel[:, 0] - self.commands[:, 0]).clip(min=0.) ) + \
                            torch.square(self.base_lin_vel[:, 1]-self.commands[:, 1])
        return torch.exp(-lin_vel_error/self.cfg.rewards.walkback_sigma)

    def _reward_posture(self):
        stand_bias = torch.zeros_like(self.dof_pos)
        stand_bias[:,1::3] += 0.2
        stand_bias[:,2::3] -= 0.3
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos - stand_bias), dim=1)

    def _reward_yawrate(self):
        return torch.square(self.base_ang_vel[:, 2]-self.commands[:, 2])   

    def _reward_alive(self):
        return torch.ones_like(self.base_ang_vel[:, 2])