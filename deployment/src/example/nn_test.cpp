/*****************************************************************
 Copyright (c) 2020, Unitree Robotics.Co.Ltd. All rights reserved.
******************************************************************/

#include "unitree_legged_sdk/unitree_legged_sdk.h"
// #include "unitree_legged_sdk/joystick.h"
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <array>
#include <vector>
#include <deque>

#include <time.h>
#include <unistd.h>

using namespace std;
using namespace UNITREE_LEGGED_SDK;

#define ACC_VARIANCE 0.1
#define SENSOR_VARIANCE 0.1
#define INITIAL_VARIANCE 0.1
#define ROBOT_TIMESTEP 0.004

std::array<float, 3> quat_rotate_inverse(std::array<float, 4> quat, std::array<float, 3> vel){
    float w; 
    float x, y, z; 
    float vx, vy, vz; 
    float a1, a2, a3, b1, b2, b3, c1, c2, c3, tmp; 
    std::array<float, 3> proj_vel; 

    w = quat[0]; 
    x = quat[1]; 
    y = quat[2]; 
    z = quat[3]; 
    vx = vel[0]; 
    vy = vel[1]; 
    vz = vel[2]; 
    tmp = 2.0 * w * w - 1.0; 
    a1 = vx * tmp;
    a2 = vy * tmp;  
    a3 = vz * tmp; 
    b1 = (y * vz - z * vy) * w * 2; 
    b2 = (z * vx - x * vz) * w * 2; 
    b3 = (x * vy - y * vx) * w * 2; 
    tmp = (x * vx + y * vy + z * vz) * 2; 
    c1 = x * tmp; 
    c2 = y * tmp; 
    c3 = z * tmp; 
    proj_vel[0] = a1 - b1 + c1; 
    proj_vel[1] = a2 - b2 + c2; 
    proj_vel[2] = a3 - b3 + c3; 
    return proj_vel; 
}

class Custom
{
public:
  Custom(uint8_t level) : safe(LeggedType::Go1),
                          udp(level, 8090, "192.168.123.10", 8007)
    {
        udp.InitCmdData(cmd);
    }
    
    void UDPRecv();
    void UDPSend();
    void RobotControl();

    // VelocityEstimator com_vel_estimator;
    std::array<float, 12> clipSafeTorque(std::array<float, 12> desired); 

    std::array<double, 3> gyros; 
    std::array<int, 3> gyro_count; 

    time_t start; 
    time_t end; 
    xRockerBtnDataStruct _keyData;

    Safety safe;
    UDP udp;
    LowCmd cmd = {0};
    LowState state = {0};

    float qInit[3]={0};
    float qDes[3]={0};
    float sin_mid_q[3] = {0.0, 1.2, -2.0};
    float Kp[3] = {0};  
    float Kd[3] = {0};
    double time_consume = 0;
    int rate_count = 0;
    int sin_count = 0;
    int motiontime = 0;
    float dt = ROBOT_TIMESTEP;     // 0.001~0.01
    float torque_min = -33.5; 
    float torque_max = 33.5; 
    std::array<float, 12> base_pos; 
    std::array<float, 12> desired_pos; 
    std::array<float, 12> dof_class_sin; 
    std::array<float, 12> dof_class_cos; 
    std::array<float, 12> last_action; 
    std::array<float, 12> action; 
    std::array<int, 12> sequence; 

    std::array<float, 3> grav; 
    std::array<float, 3> base_vel; 

    std::deque<std::array<float, 3>> gyroscope_buffer;
    std::array<float, 3> acc_bias;
    std::array<float, 4> foot_threshold;

    float lx = 0; 
    float ly = 0; 
    float rx = 0; 
    
};

void Custom::UDPRecv()
{  
    std::cout << "ENTERING UDP recv." << std::endl;
    udp.Recv();
    std::cout << "FINISHING UDP recv." << std::endl;
}

void Custom::UDPSend()
{  
    std::cout << "ENTERING UDP send." << std::endl;
    udp.Send();
    std::cout << "FINISHING UDP send." << std::endl;
    
}

double jointLinearInterpolation(double initPos, double targetPos, double rate)
{
    double p;
    rate = std::min(std::max(rate, 0.0), 1.0);
    p = initPos*(1-rate) + targetPos*rate;
    return p;
}

std::array<float, 12> Custom::clipSafeTorque(std::array<float, 12> desired){
    int i; 
    float clip_min, clip_max; 
    for (i = 0; i < 12; i ++){
        clip_min = (torque_min + 0.5 * state.motorState[i].dq) / 40 + state.motorState[i].q; 
        clip_max = (torque_max + 0.5 * state.motorState[i].dq) / 40 + state.motorState[i].q; 
        if (desired[i] < clip_min) desired[i] = clip_min; 
        if (desired[i] > clip_max) desired[i] = clip_max; 
    }
    return desired; 
}

void Custom::RobotControl()
{
  motiontime++;
  udp.GetRecv(state);
  // gravity compensation
  cmd.motorCmd[FR_0].tau = -0.65f;
  cmd.motorCmd[FL_0].tau = +0.65f;
  cmd.motorCmd[RR_0].tau = -0.65f;
  cmd.motorCmd[RL_0].tau = +0.65f;

  if (motiontime >= 500)
  {
    float torque = (0 - state.motorState[FR_1].q) * 10.0f + (0 - state.motorState[FR_1].dq) * 1.0f;
    if (torque > 5.0f)
      torque = 5.0f;
    if (torque < -5.0f)
      torque = -5.0f;

    cmd.motorCmd[FR_1].q = PosStopF;
    cmd.motorCmd[FR_1].dq = VelStopF;
    cmd.motorCmd[FR_1].Kp = 0;
    cmd.motorCmd[FR_1].Kd = 0;
    cmd.motorCmd[FR_1].tau = torque;
  }
  int res = safe.PowerProtect(cmd, state, 1);
  if (res < 0)
    exit(-1);

  udp.SetSend(cmd);
}
// void Custom::RobotControl() 
// {   
//     start = clock(); 
//     motiontime++;
//     udp.GetRecv(state);
//     int i;
//     if (motiontime <= 100){
//         for (i = 0; i < 12; i++){
//             base_pos[i] = state.motorState[i].q; 
//         }
//     }

//     else{
//         if (motiontime < 500){
//             for (i = 0; i < 12; i++ ){
//                 action[i] = base_pos[i] + (desired_pos[i] - base_pos[i]) * (motiontime - 100) / 400; 
//             }
//         }
//         else if (motiontime < 700){
//             for (i = 0; i < 12; i++ ){
//                 action[i] = desired_pos[i]; 
//             }
//             // com_vel_estimator.reset();
//         }
//         else if (motiontime < 900){
//             acc_bias[0] += state.imu.accelerometer[0] / 200;
//             acc_bias[1] += state.imu.accelerometer[1] / 200;
//             acc_bias[2] += state.imu.accelerometer[2] / 200;
//             for (int i=0; i<4;i++) {
//                 foot_threshold[i] += state.footForce[i] / 200.0;
//             }
//         }
        

//         cout << endl; 
//         for (i = 0; i < 12; i ++){
//             cmd.motorCmd[i].q = action[i]; 
//             cmd.motorCmd[i].dq = 0; 
//             cmd.motorCmd[i].Kp = 20; 
//             cmd.motorCmd[i].Kd = 0.5; 
//         }
//     }
    
//     if(motiontime > 10){
//         safe.PositionLimit(cmd);
//         safe.PowerProtect(cmd, state, 5);
//         // You can uncomment it for position protection
//         // int res2 = safe.PositionProtect(cmd, state, 0.087);
//         // if(res1 < 0) exit(-1);
//     }
//     end = clock(); 
//     cout << "time cost: " << float(end - start) / CLOCKS_PER_SEC << endl; 

//     udp.SetSend(cmd);

// }


int main(void)
{


    std::cout << "Communication level is set to LOW-level." << std::endl
              << "WARNING: Make sure the robot is hung up." << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();
    
    Custom custom(LOWLEVEL);
    // InitEnvironment();

    LoopFunc loop_control("control_loop", custom.dt, boost::bind(&Custom::RobotControl, &custom));
    LoopFunc loop_udpSend("udp_send", custom.dt, 3, boost::bind(&Custom::UDPSend, &custom));
    LoopFunc loop_udpRecv("udp_recv", custom.dt, 3, boost::bind(&Custom::UDPRecv, &custom));

    loop_udpSend.start();
    loop_udpRecv.start();
    loop_control.start();
        
    // LoopFunc loop_control("control_loop", custom.dt,    boost::bind(&Custom::RobotControl, &custom));
    // LoopFunc loop_udpSend("udp_send",     custom.dt, 3, boost::bind(&Custom::UDPSend,      &custom));
    // LoopFunc loop_udpRecv("udp_recv",     custom.dt, 3, boost::bind(&Custom::UDPRecv,      &custom));
    // std::cout << "before start" << std::endl;
    // loop_udpSend.start();
    // loop_udpRecv.start();
    // loop_control.start();
    // std::cout << "after start" << std::endl;

    while(1){
        sleep(10);
    };

    return 0; 
}
