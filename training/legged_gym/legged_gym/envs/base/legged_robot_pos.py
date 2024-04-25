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
from .legged_robot_pos_config import LeggedRobotPosCfg

class LeggedRobotPos(LeggedRobot):
    cfg : LeggedRobotPosCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()
        # Additionally initialize timer_left
        self.timer_left = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) * self.cfg.env.episode_length_s
        self.timer_left.uniform_(self.cfg.env.episode_length_s - self.cfg.domain_rand.randomize_timer_minus, self.cfg.env.episode_length_s)
        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.position_targets = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.heading_targets = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
    
    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        if self.cfg.domain_rand.randomize_dof_bias:
            self.dof_bias[env_ids] = self.dof_bias[env_ids].uniform_(-self.cfg.domain_rand.max_dof_bias, self.cfg.domain_rand.max_dof_bias)
        if self.cfg.domain_rand.erfi:
            self.erfi_rnd[env_ids] = self.erfi_rnd[env_ids].uniform_(0., 1.)
        if self.cfg.asset.load_dynamic_object:
            self.obj_state_rand[env_ids] = self.obj_state_rand[env_ids].uniform_(0., 1.)
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.contact_filt[env_ids] = False
        self.last_contacts[env_ids] = False
        # reset timer
        self.timer_left[env_ids] = -self.cfg.domain_rand.randomize_timer_minus * torch.rand(len(env_ids), device=self.device) + self.cfg.env.episode_length_s
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        
    def _update_terrain_curriculum(self, env_ids):
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.position_targets[env_ids, :2], dim=1)
        move_up = distance < self.cfg.rewards.position_target_sigma_tight
        move_down = distance > self.cfg.rewards.position_target_sigma_soft
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
        self.timer_left -= self.dt

        pos_diff = self.position_targets - self.root_states[:, 0:3]
        self.commands[:, :2] = quat_rotate_inverse(yaw_quat(self.base_quat[:]), pos_diff)[:, :2] # only x, y used here

        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[:, 2] = wrap_to_pi(self.heading_targets[:,0] - heading)

        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact

        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = (self.timer_left <= 0) # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def _resample_commands(self, env_ids):
        tbd_envs = env_ids.clone()
        while len(tbd_envs) > 0:
            _target_pos1 = torch_rand_float(self.command_ranges["pos_1"][0], self.command_ranges["pos_1"][1], (len(tbd_envs), 1), device=self.device)
            _target_pos2 = torch_rand_float(self.command_ranges["pos_2"][0], self.command_ranges["pos_2"][1], (len(tbd_envs), 1), device=self.device)
            _target_heading = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(tbd_envs), 1), device=self.device)
            if self.cfg.commands.ranges.use_polar:
                self.position_targets[tbd_envs, 0:1] = self.env_origins[tbd_envs, 0:1] + _target_pos1 * torch.cos(_target_pos2)
                self.position_targets[tbd_envs, 1:2] = self.env_origins[tbd_envs, 1:2] + _target_pos1 * torch.sin(_target_pos2)
            else:
                self.position_targets[tbd_envs, 0:1] = self.env_origins[tbd_envs, 0:1] + _target_pos1
                self.position_targets[tbd_envs, 1:2] = self.env_origins[tbd_envs, 1:2] + _target_pos2
            self.position_targets[tbd_envs, 2] = self.env_origins[tbd_envs,2] + 0.5

            pos_diff = self.position_targets[tbd_envs] - self.root_states[tbd_envs, 0:3]
            self.heading_targets[tbd_envs, :] = wrap_to_pi(_target_heading + torch.atan2(pos_diff[:,1:2],pos_diff[:,0:1]))

            self.commands[tbd_envs, :2] = quat_rotate_inverse(yaw_quat(self.base_quat[tbd_envs]), pos_diff)[:, :2] # only x, y used here
            forward = quat_apply(self.base_quat[tbd_envs], self.forward_vec[tbd_envs])
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[tbd_envs, 2] = wrap_to_pi(self.heading_targets[tbd_envs,0] - heading)
            
            # reject sampling
            in_obst = torch.zeros_like(self.heading_targets[:,0], dtype=torch.bool)
            if self.cfg.asset.load_dynamic_object:
                for _obj in range(self.cfg.asset.object_num):
                    _dist = torch.norm(self.position_targets[tbd_envs, 0:2] - self.root_states_obj[_obj][tbd_envs, 0:2], dim=-1)
                    _obj_type = _obj % (len(self.object_asset_list))
                    _radius_obj = self.object_size_list[_obj_type]
                    _radius_thr = _radius_obj * 1.1 + 0.31415926
                    in_obst[tbd_envs] = torch.logical_or(in_obst[tbd_envs], _dist<_radius_thr)
            tbd_envs = in_obst.nonzero(as_tuple=False).flatten()

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
        noise_vec[10:14] = 0. # commands
        noise_vec[14:26] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[26:38] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[38:50] = 0. # previous actions
        if self.cfg.sensors.ray2d.enable:
            noise_vec[50:] = noise_scales.ray2d * noise_level * self.obs_scales.ray2d
        return noise_vec

    def compute_cost(self):
        """ Computes cost for PPO Lagrangian
        """
        collision = self._reward_collision()
        termination = self._reward_termination()
        feet_collision = self._reward_feet_collision()
        # import pdb; pdb.set_trace()
        # self.extras['cost'] = ((collision + feet_collision) > 0).float()
        self.extras['cost'] = ((collision + termination + feet_collision) > 0).float()

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
                                    self.commands[:, :3], # 10:13
                                    self.timer_left.unsqueeze(1) / self.max_episode_length_s,  # 13:14
                                    (self.dof_pos - self.default_dof_pos - self.dof_bias) * self.obs_scales.dof_pos, # 14:26
                                    self.dof_vel * self.obs_scales.dof_vel, # 26:38
                                    self.actions # 38:50
                                    ),dim=-1)  # append ray2d obs after this, 50:
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
            # should I clip ray obs here after noise? maybe not.

    def _draw_debug_vis(self):
        super()._draw_debug_vis()
        sphere_geom1 = gymutil.WireframeSphereGeometry(0.1, 4, 4, None, color=(1, 0.2, 1))
        sphere_geom2 = gymutil.WireframeSphereGeometry(0.06, 4, 4, None, color=(1, 0.2, 1))
        for i in range(self.num_envs):
            x = self.position_targets[i,0]
            y = self.position_targets[i,1]
            z = self.position_targets[i,2]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom1, self.gym, self.viewer, self.envs[i], sphere_pose) 
            x = self.position_targets[i,0] + 0.15*torch.cos(self.heading_targets[i])
            y = self.position_targets[i,1] + 0.15*torch.sin(self.heading_targets[i])
            z = self.position_targets[i,2]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom2, self.gym, self.viewer, self.envs[i], sphere_pose) 

    #### rewards
    def _command_duration_mask(self, duration):
        mask = self.timer_left <= duration 
        return mask / duration

    def _reward_reach_pos_target_soft(self):
        distance = torch.norm(self.position_targets[:, :2] - self.root_states[:, :2], dim=1)
        return (1. /(1. + torch.square(distance / self.cfg.rewards.position_target_sigma_soft))) * self._command_duration_mask(self.cfg.rewards.rew_duration)

    def _reward_reach_pos_target_tight(self):
        distance = torch.norm(self.position_targets[:, :2] - self.root_states[:, :2], dim=1)
        return (1. /(1. + torch.square(distance / self.cfg.rewards.position_target_sigma_tight))) * self._command_duration_mask(self.cfg.rewards.rew_duration/2)
    
    def _reward_reach_heading_target(self):
        distance = torch.norm(self.position_targets[:, :2] - self.root_states[:, :2], dim=1)
        near_goal = (distance < self.cfg.rewards.position_target_sigma_soft)
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading_angle = torch.atan2(forward[:, 1], forward[:, 0])
        angle_difference = torch.abs(wrap_to_pi(heading_angle - self.heading_targets[:,0]))
        heading_rew = 1. /(1. + torch.square(angle_difference / self.cfg.rewards.heading_target_sigma))
        return heading_rew * near_goal * self._command_duration_mask(self.cfg.rewards.rew_duration)  # feel the heading rew in advance

    def _reward_reach_pos_target_times_heading(self):
        # Compute distance between robot and target positions
        distance = torch.norm(self.position_targets[:, :2] - self.root_states[:, :2], dim=1)
        
        # Compute heading angle of the robot
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading_angle = torch.atan2(forward[:, 1], forward[:, 0])
        
        # Compute angle difference between heading and positive x-axis direction
        angle_difference = torch.abs(wrap_to_pi(heading_angle - self.heading_targets[:,0]))  # 0 radians represents positive x-axis direction
        
        # Apply a penalty if the robot deviates from the positive x-axis direction
        heading_penalty = torch.abs(torch.cos(angle_difference)) # avoid negative rewards
        
        # Compute the reward based on distance and heading penalty
        distance_reward = (1. / (1. + torch.square(distance / self.cfg.rewards.position_target_sigma)))
        
        # Combine distance reward and heading penalty
        combined_reward = distance_reward * heading_penalty * self._command_duration_mask(self.cfg.rewards.rew_duration)
        return combined_reward


    def _reward_velo_dir(self):
        forward = quat_apply(self.base_quat, self.forward_vec)
        xy_dif = self.position_targets[:,:2] - self.root_states[:, :2]
        xy_dif = xy_dif / (0.001 + torch.norm(xy_dif, dim=1).unsqueeze(1))
        good_dir = forward[:,0] * xy_dif[:,0] + forward[:,1] * xy_dif[:,1] > -0.25  # base orientation -> target
        
        distance = torch.norm(self.position_targets[:, :2] - self.root_states[:, :2], dim=1)
        _rew = self.base_lin_vel[:,0].clip(min=0.0) * good_dir * (distance>self.cfg.rewards.position_target_sigma_tight) / 4.5 \
                                            + 1.0 * (distance<self.cfg.rewards.position_target_sigma_tight)
        return _rew

    def _reward_stand_still_pos(self):
        # Penalize motion at zero commands
        distance = torch.norm(self.position_targets[:, :2] - self.root_states[:, :2], dim=1)
        stand_bias = torch.zeros_like(self.dof_pos)
        stand_bias[:,1::3] += 0.2
        stand_bias[:,2::3] -= 0.3
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos - stand_bias), dim=1) * self._command_duration_mask(self.cfg.rewards.rew_duration/2) \
                                                                                             * (distance < self.cfg.rewards.position_target_sigma_tight)

    def _reward_fly(self):
        fly = torch.sum(self.contact_filt.float(), dim=-1) < 0.5
        return fly * 1.0 * (self.episode_length_buf * self.dt > 0.5)  # ignore falling down when respawned

    def _reward_termination(self):
        # Terminal reward / penalty; 5x penalty for dying in the rew_duration near goal
        distance = torch.norm(self.position_targets[:, :2] - self.root_states[:, :2], dim=1)
        return (self.reset_buf * ~self.time_out_buf) * (1.0 + \
                    4.0 * self._command_duration_mask(self.cfg.rewards.rew_duration/2) * (distance < self.cfg.rewards.position_target_sigma_tight))

    def _reward_nomove(self):
        # travel_dist = torch.norm(self.root_states[:, :2] - self.env_origins[:, :2], dim=1)
        static = torch.logical_and(torch.norm(self.base_lin_vel[:,:2], dim=-1) < 0.1, torch.abs(self.base_ang_vel[:,2]) < 0.1)

        forward = quat_apply(self.base_quat, self.forward_vec)
        xy_dif = self.position_targets[:,:2] - self.root_states[:, :2]
        xy_dif = xy_dif / (0.001 + torch.norm(xy_dif, dim=1).unsqueeze(1))
        bad_dir = forward[:,0] * xy_dif[:,0] + forward[:,1] * xy_dif[:,1] < -0.25  # base orientation not -> target

        distance = torch.norm(self.position_targets[:, :2] - self.root_states[:, :2], dim=1)

        return static * bad_dir * 1.0 * (distance > self.cfg.rewards.position_target_sigma_soft)