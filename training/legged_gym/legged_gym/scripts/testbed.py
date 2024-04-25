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
import torch.nn as nn
import time

RECORD_FRAMES = False
MOVE_CAMERA = False

one_trial = False # if True, the robot will not reset after falling
train_ra = False
test_ra = False
visualize_ra = False # there can be bugs here, not encouraged to enable.


difficulty = 2  # 0: easy; 1: medium, 2: hard
init_obst_xy = [[-3., 8., -2.5, 2.5], [-3., 8., -2.5, 2.5], [1.5, 7., -2., 2.]]  # xmin, xmax, ymin, ymax, for easy/medium/hard

def get_pos_integral(twist, tau):
    # taylored as approximation
    vx, vy, wz = twist[...,0], twist[...,1], twist[...,2]
    theta = wz * tau
    x = vx * tau - 0.5 * vy * wz * tau * tau
    y = vy * tau + 0.5 * vx * wz * tau * tau
    return x, y, theta
# twist iter params
twist_tau = 0.05
twist_eps = 0.05
twist_lam = 10.
twist_lr = 0.5
twist_min = torch.tensor([-1.5,-0.3,-3.0]).cuda()
twist_max = -twist_min

def _clip_grad(grad, thres):
    grad_norms = grad.norm(p=2, dim=-1).unsqueeze(-1) #(n,1)
    return grad * thres / torch.maximum(grad_norms, thres*torch.ones_like(grad_norms))

def sample_obstacle_test(xmin, xmax, ymin, ymax, n_env, n_obj, safedist=0.75):
    """
    this function samples obstacles in an effective distribution (coz tons of obsts at the same place make no sense)
    """
    assert ymax - ymin > 2*safedist
    assert xmax > xmin + 0.1
    obj_pos_sampled = torch.zeros(n_env,2,n_obj).to("cuda")
    # step 1
    nodes = torch.zeros(n_env,2,4).to("cuda")
    nodes[:,0,0] = xmin # x0 = xmin
    nodes[:,0,1] = xmin * 0.67 + xmax * 0.33 # x1
    nodes[:,0,2] = xmin * 0.33 + xmax * 0.67 # x2
    nodes[:,0,3] = xmax # x3 = xmax
    nodes[:,1,0] = ymin * 0.5 + ymax * 0.5 # y0 in middle
    nodes[:,1,3] = ymin * 0.5 + ymax * 0.5 # y3 in middle
    nodes[:,1,1:3] = nodes[:,1,1:3].uniform_(ymin+safedist, ymax-safedist)
    # step 2
    A = torch.stack([nodes[:,0,:]**3, nodes[:,0,:]**2, nodes[:,0,:], torch.ones_like(nodes[:,0,:])], dim=2)
    coefficients = torch.linalg.lstsq(A, nodes[:,1,:].unsqueeze(2)).solution  # (n,4,1)
    # step 3
    obj_pos_sampled[:,0,:] = obj_pos_sampled[:,0,:].uniform_(xmin, xmax) #(n,4)
    obj_pos_sampled[:,1,:] = obj_pos_sampled[:,1,:].uniform_(ymin, ymax)
    # step 4
    y_curve = coefficients[:,0] * obj_pos_sampled[:,0,:]**3 + coefficients[:,1] * obj_pos_sampled[:,0,:]**2 \
                + coefficients[:,2] * obj_pos_sampled[:,0,:] + coefficients[:,3] #(n,4)
    diffy = obj_pos_sampled[:,1,:] - y_curve
    diffy[diffy==0.] = 0.001
    obj_pos_sampled[:,1,:] = obj_pos_sampled[:,1,:] * (torch.abs(diffy)>=safedist) + (torch.sign(diffy)*safedist + y_curve) * (torch.abs(diffy)<safedist)
    return obj_pos_sampled

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    train_ra = args.trainRA
    test_ra = args.testRA
    print('train RA:', train_ra, ';test RA:', test_ra)
    global difficulty
    if train_ra: difficulty = 1
    # override some parameters for testing
    env_cfg.env.num_envs = args.num_envs
    env_cfg.terrain.num_rows = 3
    env_cfg.terrain.num_cols = 3
    env_cfg.terrain.curriculum = False
    # env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.terrain_types = ['flat']  # do not duplicate!
    env_cfg.terrain.terrain_proportions = [0.5]
    env_cfg.noise.add_noise = True
    env_cfg.noise.noise_level = 0.0 # allow illusion
    env_cfg.domain_rand.randomize_friction = True
    if train_ra: env_cfg.domain_rand.friction_range = [-0.2, -0.2]
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.max_push_vel_xy = 0.0
    env_cfg.domain_rand.randomize_dof_bias = False
    env_cfg.domain_rand.erfi = False

    # env_cfg.domain_rand.randomize_yaw = False
    # env_cfg.domain_rand.randomize_xy = False
    env_cfg.domain_rand.init_yaw_range = [-3.14159,3.14159]
    env_cfg.domain_rand.init_x_range = [-0.1,0.1]
    env_cfg.domain_rand.init_y_range = [-0.1,0.1]
    if test_ra: 
        env_cfg.domain_rand.randomize_timer_minus = 0.0
        env_cfg.asset.object_files = {'{LEGGED_GYM_ROOT_DIR}/resources/objects/cylindar.urdf': 0.415} # slightly compensate for noise
        # env_cfg.env.episode_length_s += 3
    env_cfg.init_state.pos = [0.0, 0.0, 0.32]
    env_cfg.domain_rand.init_dof_factor = [1.0,1.0]
    env_cfg.domain_rand.stand_bias3 = [0.0, 0.2, -0.3]

    if difficulty == 0:
        print('life on easy mode')
        env_cfg.asset.object_num = 3
    if difficulty == 1:
        print('difficulty: normal')
    if difficulty == 2:
        print('difficulty: hard')

    env_cfg.asset.test_mode = True
    env_cfg.asset.test_obj_pos = [[[1.5, 1.5, 2.6,3.0,3.7,4.3,5.2,5.8],[-1.0,1.0,-0.7,0.8,-0.85,-0.3,1.1,0.45]],]
    env_cfg.asset.test_obj_pos = torch.Tensor(env_cfg.asset.test_obj_pos).to("cuda")
    env_cfg.asset.test_obj_pos = env_cfg.asset.test_obj_pos[:,:,:env_cfg.asset.object_num]
    env_cfg.asset.test_obj_pos = env_cfg.asset.test_obj_pos.repeat(env_cfg.env.num_envs,1,1)
    if train_ra:
        env_cfg.asset.test_obj_pos[1:,0,:] = env_cfg.asset.test_obj_pos[1:,0,:].uniform_(init_obst_xy[difficulty][0],init_obst_xy[difficulty][1])
        env_cfg.asset.test_obj_pos[1:,1,:] = env_cfg.asset.test_obj_pos[1:,1,:].uniform_(init_obst_xy[difficulty][2],init_obst_xy[difficulty][3])
    else:
        env_cfg.asset.test_obj_pos[1:] = sample_obstacle_test(init_obst_xy[difficulty][0], init_obst_xy[difficulty][1], \
                                            init_obst_xy[difficulty][2], init_obst_xy[difficulty][3], env_cfg.env.num_envs-1, env_cfg.asset.object_num)
    # safe init
    env_cfg.asset.test_obj_pos[env_cfg.asset.test_obj_pos==0.] = 0.01
    _too_near = (env_cfg.asset.test_obj_pos.norm(dim=1) < 1.1).unsqueeze(1)
    env_cfg.asset.test_obj_pos[:,0:1,:] += _too_near * torch.sign(env_cfg.asset.test_obj_pos[:,0:1,:]) * 0.9
    env_cfg.asset.test_obj_pos[:,1:,:] += _too_near * torch.sign(env_cfg.asset.test_obj_pos[:,1:,:]) * 0.9

    env_cfg.commands.ranges.use_polar = False
    env_cfg.commands.ranges.pos_1 = [6.0,7.5]
    env_cfg.commands.ranges.pos_2 = [-1.5,1.5]
    env_cfg.commands.ranges.heading = [0.0,0.0]

    env_cfg.asset.terminate_after_contacts_on = ["base", 'FL_thigh', "FL_calf", "FR_thigh", "FR_calf"]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    def new_check_termination():  # happens before update of obs and also the relpos, filter collisions falsely detected
        env.reset_buf = torch.any(torch.norm(env.contact_forces[:, env.termination_contact_indices, :], dim=-1) > 1., dim=1)
        hor_footforce = env.contact_forces[:, env.feet_indices[:2],0:2].norm(dim=-1)
        ver_footforce = torch.abs(env.contact_forces[:, env.feet_indices[:2],2])
        foot_hor_col = torch.any(hor_footforce > 2 * ver_footforce + 10.0, dim=-1)
        env.reset_buf |= foot_hor_col
        minobjdist = torch.cat([env.obj_relpos[_obj][:].norm(dim=-1).unsqueeze(-1) for _obj in range(env.cfg.asset.object_num)], dim=-1)
        _near_obj = torch.any(minobjdist<0.95, dim=-1)  
        _near_obj = torch.logical_and(_near_obj, env.base_lin_vel[:,:2].norm(dim=-1) > 0.5)  
        _near_obj = torch.logical_or(_near_obj, torch.norm(env.contact_forces[:, 0, :], dim=-1) > 1.) 
        env.reset_buf = torch.logical_and(env.reset_buf, _near_obj)   # filter the weird collisions from simulator that cannot be explained
        env.time_out_buf = (env.timer_left <= 0) # no terminal reward for time-outs
        env.reset_buf |= env.time_out_buf
    env.check_termination = new_check_termination
    env.debug_viz = True
    obs = env.get_observations()
    env.terrain_levels[:] = 9
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    policy_name = str(task_registry.loaded_policy_path.split('/')[-2]) + str(task_registry.loaded_policy_path.split('/')[-1])
    print('\nLoaded policy from: {}\n'.format(task_registry.loaded_policy_path))

    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    ra_vf = nn.Sequential(nn.Linear(19,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1), nn.Tanh())
    ra_vf.to('cuda')
    optimizer = torch.optim.SGD(ra_vf.parameters(), lr=0.002, momentum=0.0)

    if train_ra:
        best_metric = 999.
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'RA', policy_name[:-3] + '_ra' + '.pt')
        if os.path.isfile(path):
            _load = input('load existing value? y/n\n')
            if _load != 'n':
                ra_vf = torch.load(path)
                print('loaded value from', path)
    if test_ra:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'RA')
        RA_name = policy_name[:-3] + '_ra' + '.pt'
        ra_vf = torch.load(os.path.join(path, RA_name))
        print('loaded value from', os.path.join(path, RA_name))
        rec_policy_path = LEGGED_GYM_ROOT_DIR + r"/resources/policy/recover_v4_twist.pt"
        rec_policy = torch.jit.load(rec_policy_path).cuda()
        print('loaded recovery policy from',rec_policy_path)
        mode_running = True # if False: recovery
        
    standard_raobs_init = torch.Tensor([[0,0,0,0,0,0,6.,0]+[0.,0.,0.,1.,1.,2.,2.,1.,0.,0.,0.]]).to(env.device)
    standard_raobs_die = torch.Tensor([[5.,0,0,0,0,0,6.,0]+[-2.5]*11]).to(env.device)
    standard_raobs_turn = torch.Tensor([[0,0,0,0,0,2.0, 0.5,5.8]+[2.0]*6+[0.0]*5]).to(env.device)
    ra_obs = standard_raobs_init.clone().repeat(env.num_envs,1)

    collision = torch.zeros(env.num_envs).to(env.device).bool()

    queue_len = 1001
    batch_size = 200
    hindsight = 10
    s_queue = torch.zeros((queue_len,env.num_envs,19), device = env.device, dtype=torch.float)
    g_queue = torch.zeros((queue_len,env.num_envs), device = env.device, dtype=torch.float)
    g_hs_queue = g_queue.clone()
    g_hs_span = torch.zeros((2,env.num_envs), device = env.device, dtype=torch.int) # start and end index of the latest finished episode
    l_queue = torch.zeros((queue_len,env.num_envs), device = env.device, dtype=torch.float)
    done_queue = torch.zeros((queue_len,env.num_envs), device = env.device, dtype=torch.bool)

    alive = torch.ones_like(env.reset_buf)



    # ======= metrics begin =======
    total_n_collision, total_n_reach, total_n_timeout = 0, 0, 0
    total_n_episodic_recovery, total_n_episodic_recovery_success, total_n_episodic_recovery_fail = 0, 0, 0
    total_recovery_dist = 0
    total_recovery_timesteps = 0
    total_n_collision_when_ra_on, total_n_collision_when_ra_off = 0, 0

    episode_recovery_logging = torch.zeros(env.num_envs).to(env.device).bool()
    current_recovery_status = torch.zeros(env.num_envs).to(env.device).bool()

    total_n_done = 0
    total_episode = 0
    last_obs = obs.clone()
    last_root_states = env.root_states.clone()
    last_position_targets = env.position_targets.clone()

    episode_travel_dist = torch.zeros(env.num_envs).to(env.device)
    episode_time = torch.zeros(env.num_envs).to(env.device)

    episode_max_velo = torch.zeros(env.num_envs).to(env.device)
    episode_max_velo_dist = 0
    episode_max_velo_dist_collision = 0
    episode_max_velo_dist_reach = 0
    episode_max_velo_dist_timeout = 0

    total_travel_dist = 0
    total_time = 0

    total_reach_dist = 0
    total_time = 0

    total_reach_dist = 0
    total_collision_dist = 0
    total_timeout_dist = 0

    total_reach_time = 0
    total_collision_time = 0
    total_timeout_time = 0
    # ======= metrics end =======

    
    for i in range(300*int(env.max_episode_length)):
        current_recovery_status = torch.zeros(env.num_envs).to(env.device).bool()
        where_recovery = torch.zeros(env.num_envs).to(env.device).bool()
        if i% 1000==0 and train_ra:
            # resample the obstacles
            env.cfg.asset.test_obj_pos[:,0,:] = env.cfg.asset.test_obj_pos[:,0,:].uniform_(init_obst_xy[difficulty][0],init_obst_xy[difficulty][1])
            env.cfg.asset.test_obj_pos[:,1,:] = env.cfg.asset.test_obj_pos[:,1,:].uniform_(init_obst_xy[difficulty][2],init_obst_xy[difficulty][3])
            # safe init
            env.cfg.asset.test_obj_pos[env_cfg.asset.test_obj_pos==0.] = 0.01
            _too_near = (env.cfg.asset.test_obj_pos[:].norm(dim=1) < 1.1).unsqueeze(1)
            env.cfg.asset.test_obj_pos[:,0:1,:] += _too_near * torch.sign(env.cfg.asset.test_obj_pos[:,0:1,:]) * 0.9
            env.cfg.asset.test_obj_pos[:,1:,:] += _too_near * torch.sign(env.cfg.asset.test_obj_pos[:,1:,:]) * 0.9

        if env.do_reset and (not train_ra):
            env_cfg.asset.test_obj_pos[:] = sample_obstacle_test(init_obst_xy[difficulty][0], init_obst_xy[difficulty][1], \
                                    init_obst_xy[difficulty][2], init_obst_xy[difficulty][3], env_cfg.env.num_envs, env_cfg.asset.object_num)
            # safe init
            env.cfg.asset.test_obj_pos[env_cfg.asset.test_obj_pos==0.] = 0.01
            _too_near = (env.cfg.asset.test_obj_pos[:].norm(dim=1) < 1.1).unsqueeze(1)
            env.cfg.asset.test_obj_pos[:,0:1,:] += _too_near * torch.sign(env.cfg.asset.test_obj_pos[:,0:1,:]) * 0.9
            env.cfg.asset.test_obj_pos[:,1:,:] += _too_near * torch.sign(env.cfg.asset.test_obj_pos[:,1:,:]) * 0.9

        if one_trial:
            if i == 1: env.do_reset = False
            env.timer_left[env.timer_left<0.05] = 0.05
            print('step',i,'survive rate',alive.float().mean().item())

        actions = policy(obs.detach()) * alive.unsqueeze(1)
        
        if test_ra:
            v_pred = ra_vf(ra_obs)
            start_v = ra_vf(standard_raobs_init).mean().item()
            die_v = ra_vf(standard_raobs_die).mean().item() 
            turn_v = ra_vf(standard_raobs_turn).mean().item()

            recovery = (v_pred > -twist_eps).squeeze(-1)
            where_recovery = torch.where(torch.logical_and(recovery, ~collision))[0]
            
            if collision.sum().item() > 0 and mode_running and visualize_ra:
                print(torch.norm(env.contact_forces[0, :, :], dim=-1))
                print('nooooooo i died!')
                time.sleep(5.0)
            
            if where_recovery.shape[0] > 0:
                episode_recovery_logging[where_recovery] = True
                current_recovery_status[where_recovery] = True

                mode_running = False
                twist_iter = torch.cat([env.base_lin_vel[where_recovery, 0:2],env.base_ang_vel[where_recovery, 2:3]], dim=-1)   
                twist_iter.requires_grad=True
                for _iter in range(10):
                    twist_ra_obs = torch.cat([twist_iter[...,0:2], env.base_lin_vel[where_recovery,2:3], env.base_ang_vel[where_recovery,0:2], twist_iter[...,2:3], obs[where_recovery,10:12], obs[where_recovery,-11:]],dim=-1)
                    x_iter, y_iter, _ = get_pos_integral(twist_iter, twist_tau)
                    ra_value = ra_vf(twist_ra_obs)
                    loss_separate = twist_lam * (ra_value + 2*twist_eps).clip(min=0).squeeze(-1) + 0.02*(((x_iter-obs[where_recovery,10:11].squeeze(-1))**2) + ((y_iter-obs[where_recovery,11:12].squeeze(-1))**2))
                    loss = loss_separate.sum()
                    loss.backward()
                    twist_iter.data = twist_iter.data - twist_lr * _clip_grad(twist_iter.grad.data, 1.0)
                    twist_iter.data = twist_iter.data.clip(min=twist_min, max=twist_max)
                    twist_iter.grad.zero_()

                twist_iter = twist_iter.detach()
                obs_rec = torch.cat((obs[where_recovery,:10], twist_iter, obs[where_recovery,14:50]), dim=-1)
                actions[where_recovery] = rec_policy(obs_rec.detach())
                env.cfg.sensors.ray2d.raycolor = (1.0,0.1,0.1)
                env.do_reset = True
            else:
                mode_running = True
                env.cfg.sensors.ray2d.raycolor = (0,0.5,0.5)

        ####### the step is here #######
        obs, _, rews, dones, infos = env.step(actions.detach())
        ####### the step is upward #######
        collision = torch.any(torch.norm(env.contact_forces[:, env.termination_contact_indices, :], dim=-1) > 1., dim=1)  
        hor_footforce = env.contact_forces[:, env.feet_indices[:2],0:2].norm(dim=-1)
        ver_footforce = torch.abs(env.contact_forces[:, env.feet_indices[:2],2])
        foot_hor_col = torch.any(hor_footforce > 2 * ver_footforce + 10.0, dim=-1)
        collision = torch.logical_or(collision, foot_hor_col)
        minobjdist = torch.cat([env.last_obj_relpos[_obj][:].norm(dim=-1).unsqueeze(-1) for _obj in range(env.cfg.asset.object_num)], dim=-1)
        _near_obj = torch.any(minobjdist<0.95, dim=-1)
        _near_obj = torch.logical_and(_near_obj, env.last_base_twist[:,:2].norm(dim=-1) > 0.5)
        _near_obj = torch.logical_or(_near_obj, torch.norm(env.contact_forces[:, 0, :], dim=-1) > 1.)
        collision = torch.logical_and(collision, _near_obj)  # filter the weird collisions from simulator that cannot be explained

        where_done = torch.where(dones)[0]
        where_collision = torch.where(torch.logical_and(dones, collision))[0]
        distance_to_goal = torch.norm(last_position_targets[:,0:2] - last_root_states[:,0:2], dim=-1) # when done is true, new env root states and position targets are updated, so we need to use last ones
        where_reach = torch.where(torch.logical_and(distance_to_goal < 0.65, torch.logical_and(dones, ~collision)))[0]
        where_timeout = torch.where(torch.logical_and(distance_to_goal >= 0.65, torch.logical_and(dones, ~collision)))[0]

        not_in_goal = torch.logical_and(distance_to_goal >= 0.65, ~dones)

        total_episode += where_done.shape[0]
        total_n_done += where_done.shape[0]
        total_n_collision += where_collision.shape[0]
        total_n_reach += where_reach.shape[0]
        total_n_timeout += where_timeout.shape[0]

        # dist and time
        if i > 0:

            onestep_dist = torch.norm(env.root_states[:,0:2] - last_root_states[:,0:2], dim=-1) * (not_in_goal).float()
            episode_travel_dist += onestep_dist
            episode_time += (- (obs[:,13] - last_obs[:,13]) * (not_in_goal).float()) * env.max_episode_length_s
            episode_max_velo = torch.maximum(episode_max_velo, onestep_dist/0.02)
            total_recovery_dist += onestep_dist[where_recovery].sum().item()
            total_recovery_timesteps += where_recovery.shape[0]
            

        if where_done.shape[0] > 0:
            # recovery
            total_n_episodic_recovery += episode_recovery_logging[where_done].sum().item()
            total_n_episodic_recovery_fail += (episode_recovery_logging[where_done] * collision[where_done]).sum().item()
            total_n_episodic_recovery_success += (episode_recovery_logging[where_done] * ~collision[where_done]).sum().item()
            episode_recovery_logging[where_done] = False

            total_n_collision_when_ra_on += torch.logical_and(collision[where_done], current_recovery_status[where_done]).sum().item()
            total_n_collision_when_ra_off += torch.logical_and(collision[where_done], ~current_recovery_status[where_done]).sum().item()

            # max velocity in trajectory
            episode_max_velo_dist += episode_max_velo[where_done].sum().item()
            episode_max_velo_dist_collision += episode_max_velo[where_collision].sum().item()
            episode_max_velo_dist_reach += episode_max_velo[where_reach].sum().item()
            episode_max_velo_dist_timeout += episode_max_velo[where_timeout].sum().item()
            episode_max_velo[where_done] = 0

            # collision
            collision_dist = episode_travel_dist[where_collision].sum().item()
            collision_time = episode_time[where_collision].sum().item()
            total_collision_dist += collision_dist
            total_collision_time += collision_time
            total_travel_dist += collision_dist
            total_time += collision_time

            # reach
            reach_dist = episode_travel_dist[where_reach].sum().item()
            reach_time = episode_time[where_reach].sum().item()
            total_reach_dist += reach_dist
            total_reach_time += reach_time
            total_travel_dist += reach_dist
            total_time += reach_time

            # timeout
            timeout_dist = episode_travel_dist[where_timeout].sum().item()
            timeout_time = episode_time[where_timeout].sum().item()
            total_timeout_dist += timeout_dist
            total_timeout_time += timeout_time
            total_travel_dist += timeout_dist
            total_time += timeout_time

            episode_time[where_done] = 0
            episode_travel_dist[where_done] = 0
            episode_time[where_collision] = 0
            episode_travel_dist[where_collision] = 0

            avg_collision_dist = total_collision_dist / (total_n_collision + 1e-8)
            avg_collision_time = total_collision_time / (total_n_collision + 1e-8)
            avg_reach_dist = total_reach_dist / (total_n_reach + 1e-8)
            avg_reach_time = total_reach_time / (total_n_reach + 1e-8)
            avg_timeout_dist = total_timeout_dist / (total_n_timeout + 1e-8)
            avg_timeout_time = total_timeout_time / (total_n_timeout + 1e-8)

            avg_total_dist = total_travel_dist / (total_episode + 1e-8)
            avg_total_time = total_time / (total_episode + 1e-8)

            avg_total_velocity = avg_total_dist / avg_total_time
            avg_collision_velocity = avg_collision_dist / (avg_collision_time + 1e-8)
            avg_reach_velocity = avg_reach_dist / (avg_reach_time + 1e-8)
            avg_timeout_velocity = avg_timeout_dist / (avg_timeout_time + 1e-8)
            avg_recovery_velocity = total_recovery_dist / ((total_recovery_timesteps + 1e-8)*0.02)

        if total_episode % 1 == 0 and total_episode > 0 and (not train_ra) and where_done.shape[0] > 0:
            print("========= Episode {} =========".format(total_episode))
            print('Total Episode:                         {}'.format(total_episode))
            print('Total Collision:                       {}'.format(total_n_collision))
            print('Total Reach:                           {}'.format(total_n_reach))
            print('Total Timeout:                         {}'.format(total_n_timeout))
            print('Total Collision + Reach + Timeout:      {}'.format(total_n_collision + total_n_reach + total_n_timeout))
            print('Total Done:                            {}'.format(total_n_done))
            print('Collision Rate:                        {:.2%}'.format(total_n_collision / total_episode))
            print('Reach Rate:                            {:.2%}'.format(total_n_reach / total_episode))
            print('Timeout Rate:                          {:.2%}'.format(total_n_timeout / total_episode))
            print('Average Total Velocity:                {:.2f}'.format(avg_total_velocity))
            print('Average Collision Velocity:            {:.2f}'.format(avg_collision_velocity))
            print('Average Reach Velocity:                {:.2f}'.format(avg_reach_velocity))
            print('Average Timeout Velocity:              {:.2f}'.format(avg_timeout_velocity))
            print('Average Recovery Velocity:             {:.2f}'.format(avg_recovery_velocity))
            print('Average in-trajectory Max Velocity:    {:.2f}'.format(episode_max_velo_dist / total_episode))
            print('Average in-trajectory Max Velocity Collision: {:.2f}'.format(episode_max_velo_dist_collision / (total_n_collision + 1e-8)))
            print('Average in-trajectory Max Velocity Reach: {:.2f}'.format(episode_max_velo_dist_reach / (total_n_reach + 1e-8)))
            print('Average in-trajectory Max Velocity Timeout: {:.2f}'.format(episode_max_velo_dist_timeout / (total_n_timeout + 1e-8)))
            # Recovery
            print('Episode that activated recovery:         {}'.format(total_n_episodic_recovery))
            print('Episode that activated recovery - safe:  {}'.format(total_n_episodic_recovery_success))
            print('Episode that activated recovery - collision:    {}'.format(total_n_episodic_recovery_fail))
            print('Episode that did not activate recovery - collision: {}'.format(total_n_collision - total_n_episodic_recovery_fail))
            print('Episodic recovery activation rate:          {:.2%}'.format(total_n_episodic_recovery / total_episode))
            print('Episodic recovery success rate (end up safe): {:.2%}'.format(total_n_episodic_recovery_success / (total_n_episodic_recovery + 1e-8)))
            # Collision
            print('RA activation rate for collision moments: {:.2%}'.format(total_n_collision_when_ra_on / (total_n_collision + 1e-8)))
            print('RA deactivation rate for collision moments: {:.2%}'.format(total_n_collision_when_ra_off / (total_n_collision + 1e-8)))
  
        last_obs = obs.clone()
        last_root_states = env.root_states.clone()
        last_position_targets = env.position_targets.clone()
        
        # contact_forces refresh is before done-reset-obs so it is at "t-1"
        if one_trial: alive = torch.logical_and(alive, ~collision)

        ra_obs = torch.cat([env.base_lin_vel, env.base_ang_vel, obs[:,10:12], obs[:,-len(env.ray2d_obs[0]):]], dim=-1)

        if train_ra:
            ## ls <=0: reach target; gs >0: failure
            gs = collision.float() * 2 - 1  # 1 for collision, -1 for not collision
            ls = torch.tanh( torch.log2(torch.norm(obs[:,10:12], dim=-1) / 0.65 + 1e-8) )
            
            s_queue[:-1] = s_queue[1:].clone()
            g_queue[:-1] = g_queue[1:].clone()
            l_queue[:-1] = l_queue[1:].clone()
            done_queue[:-1] = done_queue[1:].clone()
            s_queue[-1] = ra_obs.clone()
            g_queue[-1] = gs.clone()  # note that g is obtained before done and reset and obs
            l_queue[-1] = ls.clone()
            done_queue[-1] = dones.clone()  # note that s is obtained after done and reset
            ## hindsight for lipschitz ######
            g_hs_queue[:-1] = g_hs_queue[1:].clone()
            g_hs_queue[-1] = gs.clone()
            ### calculate the span for potential hindsight lipschitz
            g_hs_span[:] -= 1
            g_hs_span[0][dones] = g_hs_span[1][dones].clone() + 1
            g_hs_span[1][dones] = queue_len - 1
            g_hs_span[0] = torch.maximum(g_hs_span[0], g_hs_span[1]-hindsight)
            g_hs_span = g_hs_span * (g_hs_span>=0)
            ### overwrite with hindsight for lipschitz if terminated witg gs > 0
            range_tensor = torch.arange(queue_len).unsqueeze(1).to(env.device) #(t,1)
            mask = (range_tensor >= g_hs_span[0:1]) & (range_tensor < g_hs_span[1:2]) #(t,n)
            new_values = gs.clone().repeat(queue_len,1) #(t,n), broadcast last frame g to every timestep
            mask = mask & (new_values>0) # and: dies at last frame
            new_values -= (g_hs_span[1:2]-range_tensor)*2/hindsight*mask ## soften the values
            g_hs_queue[mask] = new_values[mask].clone()
            
            if i > queue_len and i % 20 == 0:
                false_safe, false_reach, n_fail, n_reach, accu_loss = 0, 0, 0, 0, []
                total_n_fail, total_n_reach = torch.logical_and(g_queue[1:]>0, done_queue[1:]).sum().item(), torch.logical_and(l_queue[:-1]<=0,done_queue[1:]).sum().item()
                start_v = ra_vf(standard_raobs_init).mean().item()
                die_v = ra_vf(standard_raobs_die).mean().item() 
                turn_v = ra_vf(standard_raobs_turn).mean().item()
                weight_end = 0.0 # ignore this, all samples are equal
                gamma = 0.999999 #
                print('weight of end %.3f'%(weight_end+1), 'total_n_fail',total_n_fail,'total_n_reach',total_n_reach,'gamma', gamma)
                for _start in range(0, queue_len-1, batch_size):
                    vs_old = ra_vf(s_queue[_start:_start+batch_size]).squeeze(-1)
                    with torch.no_grad():
                        vs_new = ra_vf(s_queue[_start+1:_start+batch_size+1]).squeeze(-1) * (~done_queue[_start+1:_start+batch_size+1]) + 1.0 * done_queue[_start+1:_start+batch_size+1]
                        vs_discounted_old = gamma * torch.maximum(g_hs_queue[_start+1:_start+batch_size+1], torch.minimum(l_queue[_start:_start+batch_size],vs_new))\
                                            + (1-gamma) * torch.maximum(l_queue[_start:_start+batch_size], g_hs_queue[_start+1:_start+batch_size+1])
                    v_loss = 100*torch.mean(torch.square(vs_old - vs_discounted_old) * (1.0 + weight_end * (done_queue[_start+1:_start+batch_size+1]>0)))  # 
                    optimizer.zero_grad()
                    v_loss.backward()
                    torch.nn.utils.clip_grad_norm_(ra_vf.parameters(), 1.0)
                    optimizer.step()

                    false_safe += torch.logical_and(g_queue[_start+1:_start+batch_size+1]>0 , vs_old<=0).sum().item()
                    false_reach += torch.logical_and(l_queue[_start:_start+batch_size]<=0 , vs_old>0).sum().item()
                    n_fail += (g_queue[_start+1:_start+batch_size+1]>0).sum().item()
                    n_reach += (l_queue[_start:_start+batch_size]<=0).sum().item() 
                    accu_loss.append(v_loss.item())

                new_loss = np.mean(accu_loss)
                print('value RA loss %.4f, false safe rate %.2f in %d, false reach rate %.2f in %d, standard values init %.2f die %.2f turn %.2f, step %d'%\
                                (new_loss, false_safe/(n_fail+1e-8), n_fail, false_reach/(n_reach+1e-8), n_reach, start_v, die_v, turn_v, i), end='   \n')
                
                if false_safe/(n_fail+1e-8) < best_metric and die_v > 0.2 and start_v<-0.1 and turn_v<-0.1 and i > 3000:
                    best_metric = false_safe/(n_fail+1e-8)
                    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'RA')
                    try:
                        os.mkdir(path)
                    except:
                        pass
                    RA_name = policy_name[:-3] + '_ra' + '.pt'
                    torch.save(ra_vf, os.path.join(path, RA_name))
                    print('\x1b[6;30;42m', 'saving ra model to', os.path.join(path, RA_name), '\x1b[0m' )

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

if __name__ == '__main__':
    args = get_args()
    play(args)