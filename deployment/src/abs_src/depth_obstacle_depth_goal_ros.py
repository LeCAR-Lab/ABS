#!/usr/bin/python

import sys
import time
import math
import torch
import numpy as np
import concurrent.futures
#sys.path.append('../lib/python/amd64')
sys.path.append('../lib/python/arm64')
import robot_interface as sdk

# mocap infra
from pyvicon_datastream import tools

#for ros
import rospy
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String, Header, Float64MultiArray
import onnxruntime



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

def calculate_heading_target_inital(goal_xyz, robot_xyz):
    goal_x = goal_xyz[0]
    goal_y = goal_xyz[1]
    robot_x = robot_xyz[0]/1000
    robot_y = robot_xyz[1]/1000
    angle = math.atan2(goal_y - robot_y, goal_x - robot_x)
    #import ipdb; ipdb.set_trace()
    return angle

def circle_ray_query(x0: torch.Tensor, y0: torch.Tensor, thetas: torch.Tensor, center_circle: torch.Tensor, radius: float, min_: float=0.1, max_:float = 6.0):
    """
    x0:(n,1), y0:(n,1), thetas: (n.m,t), center: (n,2)
    for each env, compute the distances for the ray from (x0, y0) at thetas to cross the circle
    return shape (n, t)
    """
    # x0 = lidar x center: -0.15
    # y0 = 0
    # theta = arange(-pi/2, pi/2, 19) #19 points in total 
    # center (obstacles xy in robot frame
    # radius: 0.4 obstacle radius
    # min_: 0.1
    # max_: 6.0
    stheta = torch.sin(thetas) # (n,t)
    ctheta = torch.cos(thetas) # (n,t)
    xc = center_circle[:,0:1] # (n,1)
    yc = center_circle[:,1:2] # (n,1)
    d_c2line = torch.abs(stheta*xc - ctheta*yc - stheta*x0 + ctheta*y0)  #(n,t)
    d_c0_square = torch.square(xc-x0) + torch.square(yc-y0)
    d_0p = torch.sqrt(d_c0_square - torch.square(d_c2line))
    semi_arc = torch.sqrt(radius**2 - torch.square(d_c2line))
    raydist = torch.nan_to_num(d_0p - semi_arc, nan = max_).clip(min=min_, max=max_)
    check_dir = ctheta * (xc-x0) + stheta * (yc-y0)
    raydist = (check_dir > 0) * raydist + (check_dir<=0) * max_
    return raydist

def transform_global_xy_to_robot_xy(global_xy, robot_xy, robot_rotation_z):
    robot_x = robot_xy[0]
    robot_y = robot_xy[1]
    assert abs(robot_x) < 100 and abs(robot_y) < 100
    global_x = global_xy[0]
    global_y = global_xy[1]
    target_from_go1_xyz= [global_x - robot_x, global_y - robot_y]
    global_x_in_robot = target_from_go1_xyz[0] * np.cos(robot_rotation_z) + target_from_go1_xyz[1] * np.sin(robot_rotation_z)
    global_y_in_robot = - target_from_go1_xyz[0] * np.sin(robot_rotation_z) + target_from_go1_xyz[1] * np.cos(robot_rotation_z)
    return np.array([global_x_in_robot, global_y_in_robot])

def transform_global_xy_to_robot_xy(global_xy, robot_xy, robot_rotation_z):
    robot_x = robot_xy[0]
    robot_y = robot_xy[1]
    assert abs(robot_x) < 100 and abs(robot_y) < 100
    global_x = global_xy[0]
    global_y = global_xy[1]
    target_from_go1_xyz= [global_x - robot_x, global_y - robot_y]
    global_x_in_robot = target_from_go1_xyz[0] * np.cos(robot_rotation_z) + target_from_go1_xyz[1] * np.sin(robot_rotation_z)
    global_y_in_robot = - target_from_go1_xyz[0] * np.sin(robot_rotation_z) + target_from_go1_xyz[1] * np.cos(robot_rotation_z)
    return np.array([global_x_in_robot, global_y_in_robot])

def make_observation_from_lowhigh_state(low_state, last_action, timestep_50hz, lidar_obs, turn, position_xy_rotation_z_obs, goal_xyz):
        
    observation = np.zeros(61, dtype=np.float32)
    ### observation 
    # base lin vel 3dim
    # base ang vel 3dim
    # projected gravity 3dim
    # commands 3dim The first two dimensions are the position of the target in the robot frame, and the last one is a random number that can be 0.
    # timer left 1dim，Starting from 1 and going to 0, subtract 0.02/episode_length for each timestep, or you can always write it as 0.5
    # dof_pos - default_dof_pos  12dim
    # dof_vel*0.2   12dim.m
    # last action 12dim

    # lidar 19dim
    # in total 68

    # base lin vel 3dim
    # observation[0:3] = [3, 0, 0] 
    # print("high_state.velocity = {}".format(high_state.velocity))
    # observation[0:3] = high_state.velocity
    # base ang vel 3dim

    # contact obs 4dim
    foot_force = low_state.footForce
    contact_bool = np.where(np.array(foot_force) > 130, 1, -1)
    # observation[0:4] = contact_bool * 2 - 1
    observation[0] = contact_bool[1] 
    observation[1] = contact_bool[0]
    observation[2] = contact_bool[3]
    observation[3] = contact_bool[2]
 
    observation[4:7] = low_state.imu.gyroscope
    # projected gravity 3dim
    q = torch.Tensor(low_state.imu.quaternion).unsqueeze(0)
    v = torch.Tensor([0, 0, -1]).unsqueeze(0)
    observation[7:10] = quat_rotate_inverse(q, v)
    # commands

    go1_xy = position_xy_rotation_z_obs[0:2]
    goal_xy = goal_xyz[0:2]
    go1_rotation_z = position_xy_rotation_z_obs[2]
    
    # goal_rotation_z = math.atan2(go1_xy[1]-goal_xy[1], go1_xy[0]-goal_xy[0])
    # command_z = (goal_rotation_z - go1_rotation_z) 
    # if command_z > math.pi:
    #     command_z -= 2*math.pi
    # elif command_z < -math.pi:
    #     command_z += 2*math.pi
    
    goal_xy_in_robotframe = transform_global_xy_to_robot_xy(goal_xy, go1_xy, go1_rotation_z)
    # import ipdb; ipdb.set_trace()
    if turn == 'straight':
        goal_xy_in_robotframe = goal_xy_in_robotframe
        command_z = np.arctan2(goal_xy_in_robotframe[1], goal_xy_in_robotframe[0])
    elif turn == 'back_home':
        goal_xy_in_robotframe = goal_xy_in_robotframe
        command_z = np.arctan2(goal_xy_in_robotframe[1], goal_xy_in_robotframe[0])
    elif turn == 'left':
        goal_xy_in_robotframe = [2, 1.5]
        command_z = np.arctan2(goal_xy_in_robotframe[1], goal_xy_in_robotframe[0])
    elif turn == 'right':
        goal_xy_in_robotframe = [2, -1.5]
        command_z = np.arctan2(goal_xy_in_robotframe[1], goal_xy_in_robotframe[0])
    elif turn == 'turn_around':
        goal_xy_in_robotframe = [-2, 0]
        command_z = -3
    elif turn == "back":
        goal_xy_in_robotframe = [-0.5, 0]
        command_z = 0
    elif turn == 'stand':
        goal_xy_in_robotframe = [0, 0]
        command_z = 0



    distance_to_goal_scalar = np.linalg.norm(goal_xy_in_robotframe[0:2])
    scaling_factor = min(1, 5/distance_to_goal_scalar+0.01)
    observation[10:13] = [goal_xy_in_robotframe[0]*scaling_factor,
                          goal_xy_in_robotframe[1]*scaling_factor,
                          command_z]
    

    
    # timer left 1dim
    observation[13] = 0.5
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
    observation[38:50] = last_action.detach().numpy()
    # ========== lidar 19dim ==========

    observation[50:61] = np.log2(np.clip(lidar_obs, 0.1, 6))
    # hack for debugging
    # observation[50:61] = np.log2(6)


    return observation



class DepthInfo:
    def __init__(self) -> None:
        self.get_info = False
        self.lidar = np.zeros((1, 11), dtype=np.float64)
    
    def lidar_callback(self, msg):
        self.get_info = True
        self.lidar = np.array(msg.data, dtype=np.float64).reshape(1, 11)
    
    def check(self):
        if not self.get_info:
            rospy.logwarn_throttle(1, "No lidar info received")

class LinveloInfo:
    def __init__(self) -> None:
        self.get_info = False
        self.linvelo = np.zeros((1, 3), dtype=np.float64)
    
    def linvelo_callback(self, msg):
        self.get_info = True
        self.linvelo = np.array(msg.data, dtype=np.float64).reshape(1, 3)
    
    def check(self):
        if not self.get_info:
            rospy.logwarn_throttle(1, "No linvelo info received")
            
class PositionXY_RotationZ_Info:
    def __init__(self) -> None:
        self.get_info = False
        self.position_xy_rotation_z = np.zeros((1, 3), dtype=np.float64)
    
    def position_xy_rotation_z_callback(self, msg):
        self.get_info = True
        self.position_xy_rotation_z = np.array(msg.data, dtype=np.float64).reshape(1, 3)
    
    def check(self):
        if not self.get_info:
            rospy.logwarn_throttle(1, "No position_xy_rotation_z info received")
            
class MocapInfo:
    def __init__(self) -> None:
        self.get_go1_info = False
        self.get_obstacle_info = False
        self.go1_headtail_xyz = np.zeros((2, 3), dtype=np.float64)
        self.obstacle_xyz_list = np.zeros((1, 3), dtype=np.float64) # (n, 3), n is the number of obstacles
    
    def go1_callback(self, msg):
        self.get_go1_info = True
        self.go1_headtail_xyz = np.array(msg.data, dtype=np.float64).reshape(2, 3)

    def obstacle_callback(self, msg):
        self.get_obstacle_info = True
        self.obstacle_xyz_list = np.array(msg.data, dtype=np.float64).reshape(1, 3)
    
    def check(self):
        if not self.get_go1_info and not self.get_obstacle_info:
            rospy.logwarn_throttle(1, "No Mocap info received")

def get_pos_integral(twist, tau):
    # taylored as approximation
    vx, vy, wz = twist[...,0], twist[...,1], twist[...,2]
    theta = wz * tau
    x = vx * tau - 0.5 * vy * wz * tau * tau
    y = vy * tau + 0.5 * vx * wz * tau * tau
    return x, y, theta


def _clip_grad(grad, thres):
    """_clip_grad(twist_iter.grad.data, 1.0) returns row-wise clipped grad"""
    grad_norms = grad.norm(p=2, dim=-1).unsqueeze(-1) #(n,1)
    return grad * thres / torch.maximum(grad_norms, thres*torch.ones_like(grad_norms))

if __name__ == '__main__':
        # VELOCIMETER
    VELOCIMETER = {"ZED": []}
    USE_RA = True
    # TWIST ITER PARAMS
    twist_tau = 0.05
    twist_eps = 0.05
    twist_lam = 10.
    twist_lr = 0.5
    twist_min = torch.tensor([-1.5, -0.3, -3.0]).cpu()
    twist_max = -twist_min
    # ======================== ROS setup ========================
    rospy.init_node('dog', anonymous=True)
    rate = rospy.Rate(50) # 50hz
    # sub_depth = rospy.Subscriber("zed_depth", Image, callback_depth)
    depth_info = DepthInfo()
    linvelo_info = LinveloInfo()
    position_xy_rotation_z_info = PositionXY_RotationZ_Info()

    sub_lidar = rospy.Subscriber("zed_lidar", Float64MultiArray, depth_info.lidar_callback, queue_size=5)
    sub_linvelo = rospy.Subscriber("zed_linvelo", Float64MultiArray, linvelo_info.linvelo_callback, queue_size=5)
    sub_position_xy_rotation_z = rospy.Subscriber("zed_position_xy_rotation_z", Float64MultiArray, position_xy_rotation_z_info.position_xy_rotation_z_callback, queue_size=5)
    pub_RA = rospy.Publisher('RA_value', Float64MultiArray, queue_size=10)
    


    NUM_JOINTS = 12
    NUM_LEGS = 4
    DEF_SLEEP_TIME = 0.001
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
    
    # policy: FL, FR, RL, RR
    # unitree： FR, RL, RR, RL
    
    
    policy_to_unitreecmd = {0: 3,  1: 4,  2: 5, 
                            3: 0,  4: 1, 5: 2,
                            6: 9,  7: 10,  8: 11,
                            9: 6,  10: 7, 11: 8}
    
    PosStopF  = math.pow(10,9)
    VelStopF  = 16000.0
    HIGHLEVEL = 0xee
    LOWLEVEL  = 0xff
    dt = 0.002
    sin_count = 0

    # Path to your ONNX model file
    onnx_model_path = 'PATH_TO_YOUR_POLICY_ONNX_MODEL.onnx'
    # Load the ONNX model
    policy_session = onnxruntime.InferenceSession(onnx_model_path)
    input_name = policy_session.get_inputs()[0].name
    output_name = policy_session.get_outputs()[0].name

    
    onnx_rec_model_path = 'PATH_TO_YOUR_RECOVERY_ONNX_MODEL.onnx'
    rec_policy_session = onnxruntime.InferenceSession(onnx_rec_model_path)
    rec_input_name = rec_policy_session.get_inputs()[0].name
    rec_output_name = rec_policy_session.get_outputs()[0].name


    
    RA_model_path = 'PATH_TO_YOUR_RA_MODEL.pt'
    RA_model = torch.load(RA_model_path).cpu()

    lowudp = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
    safe = sdk.Safety(sdk.LeggedType.Go1)

    cmd = sdk.LowCmd()
    low_state = sdk.LowState()
    lowudp.InitCmdData(cmd)

    
    motiontime = 0
    cur_pose = np.zeros(12, dtype=np.float32)
    last_action = torch.zeros(12, dtype=torch.float32)
    time_idx = 0
    timestep_50hz = 1
    inference_cnt = 1
    inference_time = 0
    RA_inference_time = 0
    mocap_time = 0
    stop_sign = False

    GOAL_XYZ = [7,  0, 0]
    HOME_XYZ = [0, 0, 0]
    ROBOT_XYZ = [0,  0, 0]
    goal_xyz = GOAL_XYZ
    
    rec_iter_cnt = 1
    rec_time_total = 0
    while not rospy.is_shutdown():

        if motiontime == 2000:
            real_control_start = time.time()
        if motiontime % 1600 == 0 and motiontime>3000:
            goal_xyz = GOAL_XYZ
            timestep_50hz = 0
            heading_target_inital = calculate_heading_target_inital(goal_xyz=GOAL_XYZ, robot_xyz=ROBOT_XYZ)
        motiontime = motiontime + 1
    
        lowudp.Recv()
        lowudp.GetRecv(low_state)



        

        keydata = low_state.wirelessRemote
        if keydata[2] == 32: # L2
            turn = 'left'
        elif keydata[2] == 16: # R2
            turn = 'right'
        elif keydata[3] == 64: # Down
            turn = 'back'
        elif keydata[3] == 16: # Up
            turn = 'stand'
            goal_xyz = GOAL_XYZ
        elif keydata[3] == 1: # Aoutput_name
            turn = 'turn_around'
        elif keydata[3] == 4: # X
            turn = 'back_home'
            goal_xyz = HOME_XYZ
        else:
            turn = 'straight'
        if turn != 'stand':
            pass
        
        stop_sign = True if keydata[3] == 2 else False # B
        if stop_sign : # left remote controller
            print("EMERGENCY STOP from remote controller")
            exit(0)

        lidar_obs = depth_info.lidar
        depth_info.check()

        linvelo_obs = linvelo_info.linvelo
        linvelo_info.check()
        
        position_xy_rotation_z_obs = position_xy_rotation_z_info.position_xy_rotation_z[0]
        position_xy_rotation_z_info.check()

        obs = make_observation_from_lowhigh_state(low_state=low_state, last_action=last_action, timestep_50hz=timestep_50hz, lidar_obs=lidar_obs, turn=turn, position_xy_rotation_z_obs=position_xy_rotation_z_obs, goal_xyz=goal_xyz)

        if time_idx % 4 == 0:
            start = time.time()
            action = policy_session.run([output_name], {input_name: obs})[0]
            action = torch.Tensor(action)
            has_nan = torch.isnan(action).any().item()
            while has_nan:
                print("nan in action")
                action = policy_session.run([output_name], {input_name: obs})[0]
                action = torch.Tensor(action)
                has_nan = torch.isnan(action).any().item()
                
            end = time.time()

            inference_time += (end - start)

            
            
            if USE_RA:
                RA_start = time.time()
                
                ra_obs = torch.cat([  torch.from_numpy(linvelo_obs[0]),  # Convert to PyTorch tensor
                                        torch.from_numpy(obs[4:7]),  # Convert to PyTorch tensor
                                        torch.from_numpy(obs[10:12]),  # Convert to PyTorch tensor
                                        torch.from_numpy(obs[50:61])  # Convert to PyTorch tensor
                                    ], dim=-1).float()

                ra_value = RA_model(ra_obs.unsqueeze(0)).detach().numpy()[0][0]
                # publish ra_value signal through ros
                RA_msg = Float64MultiArray()
                RA_data = list([ra_value])
                RA_msg.data = RA_data
                pub_RA.publish(RA_msg)

                # print("ra_obs = ", ra_obs)
                # print("ra_value = ", ra_value)
                

                
                if ra_value > -twist_eps:
                    print("RA value = {}, start recovery!!!!!!!!".format(ra_value))
                    # get recovery twist for recovery RL policy
                    twist_iter = torch.zeros(3)
                    twist_iter[0:2] = torch.from_numpy(linvelo_obs[0][0:2]) # vx, vy
                    twist_iter[2] = torch.tensor(obs[6], dtype=torch.float32) # wz
                    twist_iter.requires_grad = True
                    RECOVERY_ITER_NUM = 3
                    recovery_start = time.time()
                    for _iter in range (RECOVERY_ITER_NUM):
                        rec_iter_cnt+=1
                        print("iter = {}".format(_iter))
                        print("twist_iter = {}".format(twist_iter))
                        twist_ra_obs = torch.cat([  twist_iter[0: 2],
                                                    torch.from_numpy(np.array([linvelo_obs[0][2]], dtype=np.float32)),  # Convert to PyTorch tensor
                                                    torch.from_numpy(obs[4:6]),  # Convert to PyTorch tensor
                                                    twist_iter[2].unsqueeze(0),  # Assuming twist_iter[2] is already a PyTorch tensor
                                                    torch.from_numpy(obs[10:12]),  # Convert to PyTorch tensor
                                                    torch.from_numpy(obs[50:61])  # Convert to PyTorch tensor
                                                ], dim=-1).float()
                        ra_value = RA_model(twist_ra_obs.unsqueeze(0))[0][0]
                        
                        x_iter, y_iter, _ = get_pos_integral(twist_iter, twist_tau)
                        loss_separate = twist_lam * (ra_value + 2*twist_eps).clip(min=0).squeeze(-1) + 0.02*(((x_iter-torch.Tensor(obs[10:11]).squeeze(-1))**2) + ((y_iter-torch.Tensor(obs[11:12]).squeeze(-1))**2))
                        loss = loss_separate.sum()
                        loss.backward()

                        twist_iter.data = twist_iter.data - twist_lr * _clip_grad(twist_iter.grad.data, 1.0)
                        
                        twist_iter.data = twist_iter.data.clip(min=twist_min, max=twist_max)
                        twist_iter.grad.zero_()
                    recovery_end = time.time()
                    rec_time_total += (recovery_end - recovery_start)
                    
                    

                    twist_iter = twist_iter.detach()
                    obs_rec = np.concatenate([obs[0:10], twist_iter.numpy(), obs[14:50]], axis=-1)
                    
                    #import ipdb; ipdb.set_trace()
                    print("Action before recovery = {}".format(action))
                    action = rec_policy_session.run([rec_output_name], {rec_input_name: obs_rec})[0]
                    action = torch.Tensor(action)
                    print("Action after recovery = {}".format(action))
                    

                RA_end = time.time()

                RA_inference_time += (RA_end - RA_start)
            zed_velocity = linvelo_info.linvelo
            zed_velo_norm = math.sqrt(zed_velocity[0][0]**2 + zed_velocity[0][1]**2)
            if zed_velo_norm > 0.3 and time_idx > 0:
                VELOCIMETER['ZED'].append(zed_velo_norm)
                   
            if motiontime >= 2000:
                inference_cnt += 1
                timestep_50hz += 1

                    

            if time_idx % 100 == 0 and motiontime > 2000:
                pass
                real_control_end = time.time()
                print("Avg policy inference time = {}".format(inference_time/inference_cnt))
                print("Avg RA inference time = {}".format(RA_inference_time/inference_cnt))
            
                print("Real control frequency = {}".format( 1/( (real_control_end - real_control_start)/inference_cnt) ))
                print("motion time = {}".format(motiontime))
                print("infernce_cnt = {}".format(inference_cnt))
                print("real_control_start = {}".format(real_control_start))
                print("real control end = {}".format(real_control_end))
                print("==="*5)
                print("Avg ZED velocity = {}".format(np.mean(VELOCIMETER['ZED'], axis=0)))
                print("Recovery iter num = {}, Recovery iter AVG time = {}".format(rec_iter_cnt, rec_time_total/rec_iter_cnt))
                print("==="*10)
            

        else:
            action = last_action
        
        if motiontime < 1000: # overwirte action
            action =  torch.Tensor(obs[14:26]) * (1 - (motiontime-500)/500) * 4
        elif motiontime < 1500:
            action = -(torch.Tensor(obs[14:26]) - torch.zeros(12, dtype=torch.float32)) * (1 - (motiontime-1000)/500) * 4
        elif motiontime < 2000:
            action = torch.zeros(12, dtype=torch.float32)
            for a_i in range(action.shape[0]):
                if a_i in [1, 4, 7, 10]:
                    action[a_i] = 0.2 * 4 # 0.1 rad * 4
                elif a_i in [2, 5, 8, 11]:
                    action[a_i] = -0.3 * 4 # -0.15 rad * 4
                else:
                    pass
        elif motiontime > 10000:
            pass
        else:
            pass

        for i in range(NUM_JOINTS):
            if i % 3 == 0:
                cmd.motorCmd[policy_to_unitreecmd[i]].q = torch.clip(action[i] * 0.25 + 0, -1.047, 1.047)
            elif i % 3 == 1:
                cmd.motorCmd[policy_to_unitreecmd[i]].q = torch.clip(action[i] * 0.25 + 0.8, -0.663, 2.966)
            elif i % 3 == 2:
                cmd.motorCmd[policy_to_unitreecmd[i]].q = torch.clip(action[i] * 0.25 + (-1.5), -2.721, -0.837)
            cmd.motorCmd[i].dq = 0
            cmd.motorCmd[i].Kp = 30
            cmd.motorCmd[i].Kd = 0.65
            cmd.motorCmd[i].tau = 0 # no torque control
        
        last_action = action

        safe.PositionLimit(cmd)
        safe.PowerProtect(cmd, low_state, 8)

        if motiontime == 500:
            print(obs[14:26])
            print(action)
            a = input("make sure the current pose is correct, then press enter to continue")
            if a == 'n':
                exit(0)
        if motiontime > 500:
            lowudp.SetSend(cmd)
        lowudp.Send()
        if time_idx % 4 == 3:
            rate.sleep()
        time_idx+=1
