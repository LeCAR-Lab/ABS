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

from .base_config import BaseConfig
from .legged_robot_config import LeggedRobotCfg

class LeggedRobotPosCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 236 # 76 dim + 160 
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = False # send time out information to the algorithm
        episode_length_s = 8 # episode length in seconds  # will be randomized in [s-2, s]

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 3
        class ranges:
            pos_rho = [1.0, 3.0] # min max [m]
            pos_theta = [-3.14, 3.14]  # rad
            heading = [-3.14, 3.14]

    class domain_rand:
        randomize_friction = True
        # friction_range = [0.5, 1.25]
        friction_range = [-0.4, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 2.5
        max_push_vel_xy = 0.6
        randomize_dof_bias = False
        max_dof_bias = 0.0

    class rewards:
        class scales:
            termination = -10.0
            reach_pos_target = 10.0
            reach_pos_target_times_heading = 0
            torques = -0.0002
            dof_pos_limits = -10.0
            dof_vel = -0.0005
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            dof_acc = -2.5e-7
            collision = -1.
            action_rate = -0.01

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        position_target_sigma = 0.5
        rew_duration = 1.0
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.8
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.2
            height_measurements = 2.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)