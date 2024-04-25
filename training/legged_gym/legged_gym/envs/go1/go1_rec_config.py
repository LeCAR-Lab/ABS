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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO, LeggedRobotCfgPPOLagrangian
from legged_gym.envs.base.legged_robot_pos_config import LeggedRobotPosCfg
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.go1.go1_pos_config import Go1PosRoughCfg
import numpy as np

class Go1RecRoughCfg( LeggedRobotPosCfg ):
    class env(LeggedRobotCfg.env):
        num_observations = 49  # contact4, omega3,g3, cmd3, joint36, 49 in total
        num_envs = 4096
        episode_length_s = 2 # episode length in seconds
        send_timeouts = True

    class init_state( Go1PosRoughCfg.init_state ):
        load_init = False  #TODO: support loading from saved states
        pos = [0.0, 0.0, 0.35]

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 3
        resampling_time = 100. # time before command are changed[s]
        class ranges:
            lin_vel_x = [-1.5, 1.5] # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-3., 3.]    # min max [rad/s]

    class control( LeggedRobotPosCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 30.}  # [N*m/rad]
        damping = {'joint': 0.65}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotPosCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
        name = "go1"
        foot_name = "foot"
        penalize_contacts_on = ["calf"] # collision reward
        terminate_after_contacts_on = ["base"] # termination rewrad
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter

        load_dynamic_object = False
        test_mode = False

    class terrain( LeggedRobotPosCfg.terrain ):
        terrain_types = ['flat', 'rough', 'low_obst']  # do not duplicate!
        terrain_proportions = [0.5, 0.5, 0.5]
        measure_heights = False

    class domain_rand:
        randomize_friction = True
        friction_range = [-0.4, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1.5, 1.5]
        randomize_dof_bias = True
        max_dof_bias = 0.08

        push_robots = False
        push_interval_s = 999.
        randomize_yaw = False
        randomize_roll = True
        init_roll_range = [-3.14/6, 3.14/6]
        randomize_pitch = True
        init_pitch_range = [-3.14/6, 3.14/6]
        randomize_xy = False
        randomize_velo = True
        init_vlinx_range = [-0.5, 5.5] # fast forward!
        init_vliny_range = [-0.5,0.5]
        init_vlinz_range = [-0.5,0.5]
        init_vang_range = [-1.0,1.0]
        randomize_init_dof = True
        init_dof_factor=[0.75, 1.25]
        stand_bias3 = [0.0, 0.2, -0.3]

        erfi = True
        erfi_torq_lim = 7.0/9  # per level, curriculum

    class sensors:
        class ray2d:
            enable = False
        
        class depth_cam:
            enable = False

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.2
            height_measurements = 2.0
            ray2d = 1.0
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
            ray2d = 0.2  # 2^0.2 = 1.1487

    class rewards():
        class scales():
            termination = -100.
            torques = -0.0005
            dof_pos_limits = -20.0
            dof_vel = -0.0005
            torque_limits = -20.0
            dof_vel_limits = -20.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            dof_acc = -2.0e-7
            action_rate = -0.01
            orientation = -20.0
            collision = -0.
            # task
            yawrate = -0.5
            posture = -0.1
            walkback = 10.
            alive = 5.

        soft_dof_pos_limit = 0.95
        base_height_target = 0.25
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.85
        max_contact_force = 100. # forces above this value are penalized
        # walkback_velo = -1.0  # do not use
        walkback_sigma = 0.5**2

class Go1RecRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.003
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go1_rec_rough'
        num_steps_per_env = 24
