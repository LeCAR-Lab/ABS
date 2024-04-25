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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import time
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import shutil

RECORD_FRAMES = False
MOVE_CAMERA = False

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 9  # hardcode for safety
    env_cfg.env.episode_length_s = 5  # cannot be too short for safety
    env_cfg.terrain.num_rows = 4
    env_cfg.terrain.num_cols = 4
    env_cfg.terrain.curriculum = False

    env_cfg.sensors.depth_cam.enable = True

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.max_push_vel_xy = 0.0
    env_cfg.domain_rand.randomize_dof_bias = False
    env_cfg.asset.object_files = {
        '{LEGGED_GYM_ROOT_DIR}/resources/objects/DiningChair/model.urdf': 0.4,
        '{LEGGED_GYM_ROOT_DIR}/resources/objects/OfficeChair/model.urdf': 0.4,
        '{LEGGED_GYM_ROOT_DIR}/resources/objects/cylindar.urdf': 0.4,
        }
    # this dict can be manually changed for multiple rounds of sampling.

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.debug_viz = False
    obs = env.get_observations()
    env.terrain_levels[:] = 9
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    exported_policy_name = str(task_registry.loaded_policy_path.split('/')[-2]) + str(task_registry.loaded_policy_path.split('/')[-1])
    # export policy as a jit module (used to run it from C++)

    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs/rec_cam')
    try:
        os.mkdir(log_root)
    except:
        pass
    log_folder = 'test' + str(len(os.listdir(log_root)) + 10 + 1)
    # this '10' is a shift in the filename, so you may change it and run multiple parallel ones without conflicts
    last_log_folder = 'test' + str(len(os.listdir(log_root)) + 10)
    last_success = os.path.isfile(os.path.join(log_root,last_log_folder,'label.pkl'))
    print("last recording succeed?",last_success)
    if not last_success:
        print(last_log_folder,'last recording failed, removed, record again')
        try:
            shutil.rmtree(os.path.join(log_root,last_log_folder))
            log_folder = last_log_folder
        except:
            pass

    log_folder = os.path.join(log_root, log_folder)
    os.mkdir(log_folder)
    print('created folder', log_folder)

    labels = {}
    for i in range(50*5*4):

        env.terrain_levels[:] = torch.randint_like(env.terrain_levels[:], low=0, high=10)

        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        if i % 5 == 2:
            for robot in range(env.num_envs):
                save_name = 'robot_%d_step%d'%(robot,i)
                cam_data = env.cam_obs[robot].detach().cpu().numpy()
                ray2d_label = env.ray2d_obs[robot].detach().cpu().numpy()
                labels[save_name] = ray2d_label
                np.save(log_folder+'/'+save_name+'.npy', cam_data) 
            print('timestep %d save done'%(i))
        
    import pickle 
    with open(log_folder+'/label.pkl', 'wb') as f:
        pickle.dump(labels, f)
        
    with open(log_folder+'/label.pkl', 'rb') as f:
        _labels = pickle.load(f)
        print(_labels[save_name])
        print('saved labels')

if __name__ == '__main__':
    args = get_args()
    play(args)
