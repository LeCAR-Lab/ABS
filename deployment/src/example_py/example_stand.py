#!/usr/bin/python

import sys
import time
import math
import torch
import numpy as np


sys.path.append('../lib/python/amd64')
import robot_interface as sdk


def quat_rotate_inverse(q, v):
    shape = q.shape
    #q_w = q[:, -1]
    #q_vec = q[:, :3]
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


def make_observation_from_lowhigh_state(low_state, last_action):
    observation = np.zeros(68, dtype=np.float32)
    ### observation 
    # base lin vel 3dim
    # base ang vel 3dim
    # projected gravity 3dim
    # commands 3dim 前两维是target在robot frame的位置 最后一个是随机数可以取0
    # timer left 1dim，从1开始变为0，每个timestep减去0.02/episode_length，也可以一直写成0.5
    # dof_pos - default_dof_pos  12dim
    # dof_vel*0.2   12dim
    # last action 12dim
    # lidar 19dim
    # in total 68

    # base lin vel 3dim
    observation[0:3] = [0, 0, 0] 
    # base ang vel 3dim
    observation[3:6] = low_state.imu.gyroscope
    # projected gravity 3dim
    q = torch.Tensor(low_state.imu.quaternion).unsqueeze(0)
    v = torch.Tensor([0, 0, -1]).unsqueeze(0)
    observation[6:9] = quat_rotate_inverse(q, v)
    # commands
    observation[9:12] = [10, 0, 0]
    # timer left 1dim，从1开始变为0，每个timestep减去0.02/episode_length，也可以一直写成0.5
    observation[12] = 0.5
    # dof_pos - default_dof_pos  12dim
    observation[14] = low_state.motorState[D['FL_0']].q - 0
    observation[15] = low_state.motorState[D['FL_1']].q - 0.8
    observation[16] = low_state.motorState[D['FL_2']].q - (-1.5)
    observation[17] = low_state.motorState[D['FR_0']].q - 0
    observation[18] = low_state.motorState[D['FR_1']].q - 0.8
    observation[19] = low_state.motorState[D['FR_2']].q - (-1.5)
    observation[20] = low_state.motorState[D['RL_0']].q - 0
    observation[21] = low_state.motorState[D['RL_1']].q - 0.8
    observation[22] = low_state.motorState[D['RL_2']].q - (-1.5)
    observation[23] = low_state.motorState[D['RR_0']].q - 0
    observation[24] = low_state.motorState[D['RR_1']].q - 0.8
    observation[25] = low_state.motorState[D['RR_2']].q - (-1.5)
    # dof_vel*0.2   12dim
    observation[26] = low_state.motorState[D['FL_0']].dq * 0.2
    observation[27] = low_state.motorState[D['FL_1']].dq * 0.2
    observation[28] = low_state.motorState[D['FL_2']].dq * 0.2
    observation[29] = low_state.motorState[D['FR_0']].dq * 0.2
    observation[30] = low_state.motorState[D['FR_1']].dq * 0.2
    observation[31] = low_state.motorState[D['FR_2']].dq * 0.2
    observation[32] = low_state.motorState[D['RL_0']].dq * 0.2
    observation[33] = low_state.motorState[D['RL_1']].dq * 0.2
    observation[34] = low_state.motorState[D['RL_2']].dq * 0.2
    observation[35] = low_state.motorState[D['RR_0']].dq * 0.2
    observation[36] = low_state.motorState[D['RR_1']].dq * 0.2
    observation[37] = low_state.motorState[D['RR_2']].dq * 0.2
    # last_action 12dim
    observation[37:49] = last_action.detach().numpy()
    # lidar 19dim
    observation[49:68] = np.log(6)

    return observation


if __name__ == '__main__':

    NUM_JOINTS = 12
    NUM_LEGS = 4
    DEF_SLEEP_TIME = 0.002
    JOINT_LIMIT = np.array([       # Hip, Thigh, Calf
        [-1.047,    -0.663,      -2.9],  # MIN
        [1.047,     2.966,       -0.837]  # MAX
    ])
    STAND = np.array(([
        -0.02452479861676693, 0.8545529842376709, -1.675719976425171,
        -0.02452479861676693, 0.8545529842376709, -1.675719976425171,
        -0.02452479861676693, 0.8545529842376709, -1.675719976425171,
        -0.02452479861676693, 0.8545529842376709, -1.675719976425171
    ]))

    SIT = np.array([
        -0.27805507, 1.1002517, -2.7185173,
        0.307049, 1.0857971, -2.7133338,
        -0.263221, 1.138222, -2.7211301,
        0.2618303, 1.1157601, -2.7110581
    ])

    D = {'FR_0': 0, 'FR_1': 1, 'FR_2': 2,
         'FL_0': 3, 'FL_1': 4, 'FL_2': 5,
         'RR_0': 6, 'RR_1': 7, 'RR_2': 8,
         'RL_0': 9, 'RL_1': 10, 'RL_2': 11}
    
    policy_to_unitree = {0: 3,  1: 4,  2: 5, 
                            3: 0,  4: 1, 5: 2,
                            6: 9,  7: 10,  8: 11,
                            9: 6,  10: 7, 11: 8}
    
    PosStopF  = math.pow(10,9)
    VelStopF  = 16000.0
    HIGHLEVEL = 0xee
    LOWLEVEL  = 0xff
    dt = 0.002
    sin_count = 0


    lowudp = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
    safe = sdk.Safety(sdk.LeggedType.Go1)
    #highudp = sdk.UDP(HIGHLEVEL, 8081, "192.168.123.161", 8008)

    cmd = sdk.LowCmd()
    low_state = sdk.LowState()
    lowudp.InitCmdData(cmd)

    
    motiontime = 0
    cur_pose = np.zeros(12, dtype=np.float32)
    last_action = torch.zeros(12, dtype=torch.float32)
    time_idx = 0
    # time.sleep(1)
    # lowudp.Recv()
    # lowudp.GetRecv(low_state)
    # time.sleep(0.1)
    while True:
        time.sleep(0.005)
        motiontime = motiontime + 1

        lowudp.Recv()
        lowudp.GetRecv(low_state)

        obs = make_observation_from_lowhigh_state(low_state=low_state, last_action=last_action)

        if motiontime < 2500: # overwirte action
            action =  torch.Tensor(obs[14:26]) * (1 - (motiontime-2000)/500) * 4
        elif motiontime < 3000:
            action = -(torch.Tensor(obs[14:26]) - torch.zeros(12, dtype=torch.float32)) * (1 - (motiontime-2500)/500) * 4
        elif motiontime < 3500:
            action = torch.zeros(12, dtype=torch.float32)
        else:
            exit(0)

        for i in range(NUM_JOINTS):
            if i % 3 == 0:
                cmd.motorCmd[policy_to_unitree[i]].q = torch.clip(action[i] * 0.25 + 0, -1.047, 1.047)
            elif i % 3 == 1:
                cmd.motorCmd[policy_to_unitree[i]].q = torch.clip(action[i] * 0.25 + 0.8, -0.663, 2.966)
            elif i % 3 == 2:
                cmd.motorCmd[policy_to_unitree[i]].q = torch.clip(action[i] *0.25 + (-1.5), -2.721, -0.837)
            cmd.motorCmd[i].dq = 0
            cmd.motorCmd[i].Kp = 30
            cmd.motorCmd[i].Kd = 0.65
            cmd.motorCmd[i].tau = 0 # no torque control
    
        last_action = action
        print(obs[14:26])
        print(action)
        #import ipdb; ipdb.set_trace()

            
        
        #if motiontime > 10:
            #exit(0)
            #import ipdb; ipdb.set_trace()
        safe.PositionLimit(cmd)
        safe.PowerProtect(cmd, low_state, 8)
        if motiontime == 2000:
            a = input("make sure the current pose is correct, then press enter to continue")
            if a == 'n':
                exit(0)
        if motiontime > 2000:
            lowudp.SetSend(cmd)
        lowudp.Send()
        time_idx+=1

