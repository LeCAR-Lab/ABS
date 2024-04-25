import pyzed.sl as sl
import math
import numpy as np
import sys
import math
import cv2
import time
import torch

#for ros image publish
import rospy
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String, Header, Float64MultiArray


def compute_xy_vel_in_camera_frame(vel_xy_world, world_rotation):
    vel_x_in_world = vel_xy_world[0]
    vel_y_in_world = vel_xy_world[1]
    rotation_z_in_world = world_rotation[2]

    vel_x_in_camera = vel_x_in_world * np.cos(rotation_z_in_world) + vel_y_in_world * np.sin(rotation_z_in_world)
    vel_y_in_camera = -vel_x_in_world * np.sin(rotation_z_in_world) + vel_y_in_world * np.cos(rotation_z_in_world)

    compute_xy_vel_in_camera_frame = np.array([vel_x_in_camera, vel_y_in_camera])
    return compute_xy_vel_in_camera_frame


def main():
    depth_lidar_torch_model_path = "PATH_TO_RAYPREDICTION_MODEL.pt"
    depth_lidar_torch_model = torch.jit.load(depth_lidar_torch_model_path).cuda()

    #ros init
    rospy.init_node('zed_depthimg_linvel', anonymous=True)
    pub_depth = rospy.Publisher('zed_depthimg', Image, queue_size=10)
    pub_linvelo = rospy.Publisher('zed_linvelo', Float64MultiArray, queue_size=10)
    pub_position_xy_rotation_z = rospy.Publisher('zed_position_xy_rotation_z', Float64MultiArray, queue_size=10)
    pub_lidar = rospy.Publisher('zed_lidar', Float64MultiArray, queue_size=10)
    bridge = CvBridge()



    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    init_params.depth_maximum_distance = 6       # Set the maximum depth perception distance to 40m
    # init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode
    # init_params.depth_mode = sl.DEPTH_MODE.QUALITY  # Use ULTRA depth mode
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE # Use ULTRA depth mode
    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS: #Ensure the camera has opened succesfully
        print("Camera Open : "+repr(status)+". Exit program.")
        exit()

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.enable_fill_mode = True

    i = 0
    depth = sl.Mat()

    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(2.75,4.0,0))


    tracking_params = sl.PositionalTrackingParameters() #set parameters for Positional Tracking
    tracking_params.enable_imu_fusion = True 
    tracking_params.enable_area_memory = False
    tracking_params.enable_pose_smoothing = True
    status = zed.enable_positional_tracking(tracking_params) #enable Positional Tracking
    if status != sl.ERROR_CODE.SUCCESS:
        print("Enable Positional Tracking : "+repr(status)+". Exit program.")
        zed.close()
        exit()

    camera_pose = sl.Pose()
    camera_info = zed.get_camera_information()

    py_translation = sl.Translation()
    pose_data = sl.Transform()
    last_translation = np.array([0, 0, 0])
    last_timestamp = 0

    
    depth_sensing_time_total = 0
    velo_estimate_time_total = 0
    model_inference_time_total = 0
    zed_time_total = 0
    ros_pub_time_total = 0
    time_start = time.time()
    while True:
        # A new image is available if grab() returns SUCCESS
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed_time_start = time.time()
            i += 1    
            velo_estimate_time_start = time.time()
            tracking_state = zed.get_position(camera_pose,sl.REFERENCE_FRAME.WORLD) #Get the position of the camera in a fixed reference frame (the World Frame)
            if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                #Get rotation and translation and displays it
                rotation = camera_pose.get_rotation_vector()
                translation = camera_pose.get_translation(py_translation)
                timestamp = camera_pose.timestamp.get_nanoseconds()
                
                time_now = time.time()
                FPS = i / (time_now - time_start)
                if i == 0:  
                    last_translation = np.array([translation.get()[0], translation.get()[1], translation.get()[2]])
                    last_timestamp = timestamp
                else:
                    # print(timestamp)
                    vel_xy_world = np.array([translation.get()[0] - last_translation[0], translation.get()[1] - last_translation[1]]) / ((timestamp - last_timestamp) / 1e9)
                    # import ipdb; ipdb.set_trace()
                    # print(1/FPS)
                    # print((timestamp - last_timestamp)/1e9)
                    # vel_xy_world = np.array([translation.get()[0] - last_translation[0], translation.get()[1] - last_translation[1]]) * FPS
                    vel_xy_camera = compute_xy_vel_in_camera_frame(vel_xy_world, rotation)
                    last_translation = np.array([translation.get()[0], translation.get()[1], translation.get()[2]])
                    last_timestamp = timestamp
                    # print("vel_xy_camera: ", vel_xy_camera)
            velo_estimate_time_end = time.time()
            velo_estimate_time_total += (velo_estimate_time_end - velo_estimate_time_start)

            depth_sensing_time_start = time.time()
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH, sl.MEM.CPU, sl.Resolution(1280//8, 720//8))
            depth_array = np.array(depth.get_data(), dtype=np.float32)  # Convert depth data to numpy array if it's not al 
            depth_array[np.isinf(depth_array)] = init_params.depth_maximum_distance
            depth_sensing_time_end = time.time()
            depth_sensing_time_total += (depth_sensing_time_end - depth_sensing_time_start)
            zed_time_end = time.time()
            zed_time_total += zed_time_end - zed_time_start
            
            # cv2.imshow("Depth", depth_array)
            # cv2.waitKey(1)
            model_inference_time_start = time.time()
            depth_array_for_model = torch.from_numpy(np.log2(depth_array)).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).cuda()
            depth_from_torch_model = 2**depth_lidar_torch_model(depth_array_for_model)
            model_inference_time_end = time.time()
            model_inference_time_total += model_inference_time_end - model_inference_time_start
            
            
            ros_pub_time_start = time.time()
            ros_image = bridge.cv2_to_imgmsg(depth_array, "32FC1")
            ros_image.header.stamp = rospy.Time.now()
            ros_image.header.frame_id = "zed_depth"
            pub_depth.publish(ros_image)

            linvel_msg = Float64MultiArray()
            linvel_data = list(np.array([vel_xy_camera[0], vel_xy_camera[1], 0.0]).reshape(-1))
            linvel_msg.data = linvel_data
            #print(linvel_data)
            pub_linvelo.publish(linvel_msg)
            
            position_xy_rotation_z_msg = Float64MultiArray()
            
            position_xy_rotation_z_data = list(np.array([translation.get()[0], translation.get()[1], rotation[2]]).reshape(-1))
            # position_xy_rotation_z_data = list(np.array([1, 0, 0]).reshape(-1))
            position_xy_rotation_z_msg.data = position_xy_rotation_z_data
            pub_position_xy_rotation_z.publish(position_xy_rotation_z_msg)
            print(position_xy_rotation_z_data)
            # print("position xy = {}".format(position_xy_rotation_z_data[0:2]))
            # print("rotation z = {}".format(position_xy_rotation_z_data[2]))
            
            

            lidar_msg = Float64MultiArray()
            lidar_data = list(depth_from_torch_model.reshape(-1))
            lidar_msg.data = lidar_data
            pub_lidar.publish(lidar_msg)

            if i % 100 == 0:
                print(i)
                time_sofar = time.time() - time_start
                print("Time so far: ", time_sofar)
                print("ZED FPS: ", i/zed_time_total)
                print("Depth Sensing FPS: ", i/depth_sensing_time_total)
                print("Velo Estimation FPS: ", i/velo_estimate_time_total)
                print("Depth Model FPS: ", i/model_inference_time_total)
                print("ROS Pub FPS: ", i/ros_pub_time_total)
                print("FPS: ", i/time_sofar)  
            ros_pub_time_end = time.time()
            ros_pub_time_total += ros_pub_time_end - ros_pub_time_start                                                                 

        if rospy.is_shutdown():
            break    




    zed.close()

if __name__ == "__main__":
    main()
