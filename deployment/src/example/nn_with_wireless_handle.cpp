/*****************************************************************
 Copyright (c) 2020, Unitree Robotics.Co.Ltd. All rights reserved.
******************************************************************/

#include "unitree_legged_sdk/unitree_legged_sdk.h"
#include "unitree_legged_sdk/joystick.h"
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <array>
#include <vector>
#include <deque>
#include <torch/script.h>
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
    Custom(uint8_t level, double vel_x, double vel_y, double vel_w): safe(LeggedType::Go1), udp(level, 8090, "192.168.123.10", 8007) {
        udp.InitCmdData(cmd);
        base_pos[FL_0] = 0.4; 
        base_pos[FL_1] = 1.16; 
        base_pos[FL_2] = -2.7; 
        base_pos[FR_0] = -0.4; 
        base_pos[FR_1] = 1.16; 
        base_pos[FR_2] = -2.7; 
        base_pos[RL_0] = 0.4; 
        base_pos[RL_1] = 1.16;
        base_pos[RL_2] = -2.7; 
        base_pos[RR_0] = -0.4; 
        base_pos[RR_1] = 1.16; 
        base_pos[RR_2] = -2.7; 
        sequence[0] = FL_0; 
        sequence[1] = FL_1; 
        sequence[2] = FL_2; 
        sequence[3] = FR_0; 
        sequence[4] = FR_1; 
        sequence[5] = FR_2; 
        sequence[6] = RL_0; 
        sequence[7] = RL_1; 
        sequence[8] = RL_2; 
        sequence[9] = RR_0; 
        sequence[10] = RR_1; 
        sequence[11] = RR_2; 
        desired_pos[FL_0] = 0.1; 
        desired_pos[FL_1] = 0.8; 
        desired_pos[FL_2] = -1.5; 
        desired_pos[FR_0] = -0.1; 
        desired_pos[FR_1] = 0.8; 
        desired_pos[FR_2] = -1.5; 
        desired_pos[RL_0] = 0.1; 
        desired_pos[RL_1] = 1.0; 
        desired_pos[RL_2] = -1.5; 
        desired_pos[RR_0] = 0.1; 
        desired_pos[RR_1] = 1.0; 
        desired_pos[RR_2] = -1.5; 
        // desired_pos[FL_0] = 0; 
        // desired_pos[FL_1] = 0.9; 
        // desired_pos[FL_2] = -1.5; 
        // desired_pos[FR_0] = 0; 
        // desired_pos[FR_1] = 0.9; 
        // desired_pos[FR_2] = -1.5; 
        // desired_pos[RL_0] = 0; 
        // desired_pos[RL_1] = 0.9; 
        // desired_pos[RL_2] = -1.5; 
        // desired_pos[RR_0] = 0; 
        // desired_pos[RR_1] = 0.9; 
        // desired_pos[RR_2] = -1.5; 
        dof_class_sin[FL_0] = 0; 
        dof_class_sin[FL_1] = 0.4; 
        dof_class_sin[FL_2] = 0.0; 
        dof_class_sin[FR_0] = 0; 
        dof_class_sin[FR_1] = -0.4; 
        dof_class_sin[FR_2] = 0.0; 
        dof_class_sin[RL_0] = 0; 
        dof_class_sin[RL_1] = 0.4; 
        dof_class_sin[RL_2] = 0.0; 
        dof_class_sin[RR_0] = 0; 
        dof_class_sin[RR_1] = -0.4; 
        dof_class_sin[RR_2] = 0.0; 
        dof_class_cos[FL_0] = 0; 
        dof_class_cos[FL_1] = 0.0; 
        dof_class_cos[FL_2] = 0.7; 
        dof_class_cos[FR_0] = 0; 
        dof_class_cos[FR_1] = 0.0; 
        dof_class_cos[FR_2] = -0.7; 
        dof_class_cos[RL_0] = 0; 
        dof_class_cos[RL_1] = 0.0; 
        dof_class_cos[RL_2] = 0.7; 
        dof_class_cos[RR_0] = 0; 
        dof_class_cos[RR_1] = 0.0; 
        dof_class_cos[RR_2] = -0.7; 
        // tensor.push_back(torch::rand({47})); 
        tensor.push_back(torch::rand({1})); 
        // policy = torch::jit::load("../rough_policy_penh_wlinvel.pt"); 
        // policy = torch::jit::load("../rough_policy_noisygyro.pt");

        policy = torch::jit::load("/home/tairanhe/workspace/sim2real-locomotion-deployment/unitree_legged_sdk/model/go1_apr10.pt");

        
        // policy = torch::jit::load("/home/tairanhe/workspace/sim2real-locomotion-deployment/unitree_legged_sdk/model/policy_1.pt");

        for (int i = 0; i < 100; i++){
            // tensor[0] = torch::rand({1}); 
            tensor[0] = torch::rand({48}); 
            out = policy.forward(tensor).toTensor(); 
        }

        // neural_vel_estimator = torch::jit::load("../neural_vel_predictor_scripted.pt");
        vel_input_tensor.push_back(torch::rand({39 * 2}));
        // vel_output = neural_vel_estimator.forward(vel_input_tensor).toTensor();
        
        grav[0] = 0.0; 
        grav[1] = 0.0; 
        grav[2] = -1.0; 
        base_vel[0] = 0.0; 
        base_vel[1] = 0.0; 
        base_vel[2] = 0.0; 
        for (int i = 0; i < 12; i++){
            last_action[i] = 0; //desired_pos[i]; 
        }
        // com_vel_estimator.reset();
        exp_vel_x = vel_x;
        exp_vel_y = vel_y; 
        exp_vel_w = vel_w;
        acc_bias[0] = 0.;
        acc_bias[1] = 0.;
        acc_bias[2] = 0.;
        for (int i=0; i<4; i++) foot_threshold[i] = 0;
        for (int i = 0; i < 3; i ++){
            gyros[i] = 0.0; 
            gyro_count[i] = 0; 
        }
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
    double exp_vel_x, exp_vel_y, exp_vel_w; 
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

    std::vector<torch::jit::IValue> tensor; 
    torch::Tensor out; 
    torch::Tensor tmp_obs = torch::rand(48); 
    torch::Tensor index = torch::rand(1); 
    torch::jit::script::Module policy; 

    std::vector<torch::jit::IValue> vel_input_tensor;
    torch::Tensor vel_output;
    torch::Tensor tmp_last_vel_obs = torch::zeros(39);
    torch::Tensor tmp_cur_vel_obs = torch::zeros(39);
    // torch::jit::script::Module neural_vel_estimator;

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
    udp.Recv();
}

void Custom::UDPSend()
{  
    udp.Send();
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
    start = clock(); 
    motiontime++;
    udp.GetRecv(state);
    // com_vel_estimator.update(&state);
    std::array<float, 3> tmp_gyroscope; 
    for (int i; i < 3; i ++){
        tmp_gyroscope[i] = state.imu.gyroscope[i]; 
    }
    gyroscope_buffer.push_back(tmp_gyroscope);
    memcpy(&_keyData, &state.wirelessRemote, 40);

    std::array<double, 3> mean_gyro; 

    if (gyroscope_buffer.size() > 10) {
        gyroscope_buffer.pop_front();
    }
    // printf("%d  %f\n", motiontime, state.motorState[FR_2].q);
    // cout << "motor angle:";
    // for (int i=0; i < 12; i++) {
    //     cout << state.motorState[i].q << ",";
    // }
    printf("%d  %f\n", motiontime, state.imu.quaternion[2]);
    int i; 

    std::array<float, 4> quat; // = state.imu.quaternion; 
    for (int i = 0; i < 4; i ++){
        quat[i] = state.imu.quaternion[i]; 
    }
    std::array<float, 3> proj_grav = quat_rotate_inverse(quat, grav);
    
    cout << "Foot Force: "; 
    for (i = 0; i < 4; i++){
        cout << state.footForce[i] << ", "; 
    }
    cout << endl; 

    // cout << "Est Foot Force: "; 
    // for (i = 0; i < 4; i ++){
    //     cout << state.footForceEst[i] << ", "; 
    // }
    // cout << endl; 

    std::array<float, 3> gyro_in_base = quat_rotate_inverse(quat, tmp_gyroscope);
    // std::array<float, 3> gyro_in_base = state.imu.gyroscope;
    for (i = 0; i < 3; i++){
        gyros[i] += gyro_in_base[i]; 
        gyro_count[i] ++; 
    }

    // base_vel[0] += state.imu.accelerometer[0] * dt; 
    // base_vel[1] += state.imu.accelerometer[1] * dt; 
    // base_vel[2] += state.imu.accelerometer[2] * dt; 
    // if (motiontime < 500+20) base_vel[0] = (motiontime - 500) * dt * 2 * 100 / 20; 
    // else base_vel[0] = 2; 
    // base_vel[0] = (motiontime - 500) * dt; 
    // base_vel[1] = 0; 
    // base_vel[2] = 0; 
    
    // cout << "Motor State: "; 
    // cout << state.motorState[FL_0].q << ", "; 
    // cout << state.motorState[FL_1].q << ", "; 
    // cout << state.motorState[FL_2].q << ", "; 
    // cout << state.motorState[FR_0].q << ", "; 
    // cout << state.motorState[FR_1].q << ", "; 
    // cout << state.motorState[FR_2].q << ", "; 
    // cout << state.motorState[RL_0].q << ", "; 
    // cout << state.motorState[RL_1].q << ", "; 
    // cout << state.motorState[RL_2].q << ", "; 
    // cout << state.motorState[RR_0].q << ", "; 
    // cout << state.motorState[RR_1].q << ", "; 
    // cout << state.motorState[RR_2].q << endl; 

    // gravity compensation
    // cmd.motorCmd[FR_0].tau = -0.65f;
    // cmd.motorCmd[FL_0].tau = +0.65f;
    // cmd.motorCmd[RR_0].tau = -0.65f;
    // cmd.motorCmd[RL_0].tau = +0.65f;

    if (motiontime <= 100){
        for (i = 0; i < 12; i++){
            base_pos[i] = state.motorState[i].q; 
        }
    }

    else{
        if (motiontime < 500){
            for (i = 0; i < 12; i++ ){
                action[i] = base_pos[i] + (desired_pos[i] - base_pos[i]) * (motiontime - 100) / 400; 
            }
        }
        else if (motiontime < 700){
            for (i = 0; i < 12; i++ ){
                action[i] = desired_pos[i]; 
            }
            // com_vel_estimator.reset();
        }
        else if (motiontime < 900){
            acc_bias[0] += state.imu.accelerometer[0] / 200;
            acc_bias[1] += state.imu.accelerometer[1] / 200;
            acc_bias[2] += state.imu.accelerometer[2] / 200;
            for (int i=0; i<4;i++) {
                foot_threshold[i] += state.footForce[i] / 200.0;
            }
        }
        else if (motiontime % 5 == 0){
            if (motiontime == 900){
                // com_vel_estimator.acc_bias[0] = acc_bias[0];
                // com_vel_estimator.acc_bias[1] = acc_bias[1];
                // com_vel_estimator.acc_bias[2] = acc_bias[2];
                // for (int i=0; i<4;i++) com_vel_estimator.foot_threshold[i] = 20; //foot_threshold[i];
            }
            // for (i = 0; i < 12; i++ ){
            //     action[i] = desired_pos[i]; 
            //     action[i] += cos((motiontime - 400) * 2 * M_PI / 50) * dof_class_sin[i]; 
            //     action[i] -= sin((motiontime - 400) * 2 * M_PI / 50) * dof_class_cos[i]; 
            // }
            // index.index_put_({0}, motiontime - 400); 

            for (i = 0; i < 3; i++){
                mean_gyro[i] = gyros[i] / gyro_count[i]; 
                gyros[i] = 0.0; 
                gyro_count[i] = 0; 
            }
    
            // Eigen::VectorXd est_vel = com_vel_estimator.estimated_velocity;
            // base_vel[0] = est_vel[0];
            // base_vel[1] = est_vel[1];
            // base_vel[2] = est_vel[2];
            
            // adding values to filter
            for (int i=0; i<3; i++) tmp_cur_vel_obs[i] = proj_grav[i];
            for (int i=0; i<12; i++) tmp_cur_vel_obs[3 + i] = state.motorState[sequence[i]].q;
            for (int i=0; i<12; i++) tmp_cur_vel_obs[15 + i] = state.motorState[sequence[i]].dq;
            for (int i=0; i<12; i++) tmp_cur_vel_obs[27 + i] = last_action[i];
            vel_input_tensor[0] = torch::cat({tmp_last_vel_obs, tmp_cur_vel_obs}, 0);
            // vel_output = neural_vel_estimator.forward(vel_input_tensor).toTensor();

            int start_index = 0;
            // Put lin vel
            for (i = 0; i < 3; i++){
                tmp_obs.index_put_({start_index}, base_vel[i] * 0.); 
                // tmp_obs.index_put_({start_index}, vel_output.index({i}).item().toFloat() * 2);
                start_index += 1;
            }
            for (int i=0; i<39; i++) tmp_last_vel_obs.index_put_({i}, tmp_cur_vel_obs.index({i}));

            // Put ang vel
            for (i = 0; i < 3; i++){
                // tmp_obs.index_put_({start_index}, state.imu.gyroscope[i] * 0.25);
                tmp_obs.index_put_({start_index}, mean_gyro[i] * 0.);
                start_index += 1; 
            }

            // Put proj grav
            for (i = 0; i < 3; i++){
                tmp_obs.index_put_({start_index}, proj_grav[i]); 
                start_index += 1;
            }

            // Put velocity command
            lx = lx * 0.8 + _keyData.lx * 0.2; 
            ly = ly * 0.8 + _keyData.ly * 0.2; 
            rx = rx * 0.8 + _keyData.rx * 0.2; 
            tmp_obs.index_put_({start_index}, ly * 2); 
            tmp_obs.index_put_({start_index + 1}, -lx * 2); 
            tmp_obs.index_put_({start_index + 2}, -rx);
            start_index += 3; 

            // Put dof pos
            tmp_obs.index_put_({start_index}, state.motorState[FL_0].q - 0.1); 
            tmp_obs.index_put_({start_index + 1}, state.motorState[FL_1].q - 0.8); 
            tmp_obs.index_put_({start_index + 2}, state.motorState[FL_2].q + 1.5); 
            tmp_obs.index_put_({start_index + 3}, state.motorState[FR_0].q + 0.1); 
            tmp_obs.index_put_({start_index + 4}, state.motorState[FR_1].q - 0.8); 
            tmp_obs.index_put_({start_index + 5}, state.motorState[FR_2].q + 1.5); 
            tmp_obs.index_put_({start_index + 6}, state.motorState[RL_0].q - 0.1); 
            tmp_obs.index_put_({start_index + 7}, state.motorState[RL_1].q - 1.0); 
            tmp_obs.index_put_({start_index + 8}, state.motorState[RL_2].q + 1.5); 
            tmp_obs.index_put_({start_index + 9}, state.motorState[RR_0].q + 0.1); 
            tmp_obs.index_put_({start_index + 10}, state.motorState[RR_1].q - 1.0); 
            tmp_obs.index_put_({start_index + 11}, state.motorState[RR_2].q + 1.5); 
            start_index += 12;

            // Put dof vel
            for (int i=0; i<12; i++) {
                tmp_obs.index_put_({start_index}, state.motorState[sequence[i]].dq * 0.05);
                start_index += 1;
            }
            // tmp_obs.index_put_({21+3}, state.motorState[FL_0].dq * 0.05); 
            // tmp_obs.index_put_({22+3}, state.motorState[FL_1].dq * 0.05); 
            // tmp_obs.index_put_({23+3}, state.motorState[FL_2].dq * 0.05); 
            // tmp_obs.index_put_({24+3}, state.motorState[FR_0].dq * 0.05); 
            // tmp_obs.index_put_({25+3}, state.motorState[FR_1].dq * 0.05); 
            // tmp_obs.index_put_({26+3}, state.motorState[FR_2].dq * 0.05); 
            // tmp_obs.index_put_({27+3}, state.motorState[RL_0].dq * 0.05); 
            // tmp_obs.index_put_({28+3}, state.motorState[RL_1].dq * 0.05); 
            // tmp_obs.index_put_({29+3}, state.motorState[RL_2].dq * 0.05); 
            // tmp_obs.index_put_({30+3}, state.motorState[RR_0].dq * 0.05); 
            // tmp_obs.index_put_({31+3}, state.motorState[RR_1].dq * 0.05); 
            // tmp_obs.index_put_({32+3}, state.motorState[RR_2].dq * 0.05); 

            // Put last action
            for (int i=0; i<12; i++) {
                tmp_obs.index_put_({start_index}, last_action[sequence[i]]);
                start_index += 1;
            }
            // tmp_obs.index_put_({36}, last_action[FL_0]); 
            // tmp_obs.index_put_({37}, last_action[FL_1]); 
            // tmp_obs.index_put_({38}, last_action[FL_2]); 
            // tmp_obs.index_put_({39}, last_action[FR_0]); 
            // tmp_obs.index_put_({40}, last_action[FR_1]); 
            // tmp_obs.index_put_({41}, last_action[FR_2]); 
            // tmp_obs.index_put_({42}, last_action[RL_0]); 
            // tmp_obs.index_put_({43}, last_action[RL_1]); 
            // tmp_obs.index_put_({44}, last_action[RL_2]); 
            // tmp_obs.index_put_({45}, last_action[RR_0]); 
            // tmp_obs.index_put_({46}, last_action[RR_1]); 
            // tmp_obs.index_put_({47}, last_action[RR_2]); 

            // Put time info
            // tmp_obs[45-3] = motiontime - 400; 

            // cout << "input observation: "; 
            // for (i = 0; i < 48; i ++){
            //     cout << tmp_obs.index({i}).item().toFloat() << ", "; 
            // }
            // cout << endl; 

            tensor[0] = tmp_obs; 
            out = policy.forward(tensor).toTensor(); 
            action[FR_0] = out.index({3}).item().toFloat() * 0.25 - 0.1; 
            action[FR_1] = out.index({4}).item().toFloat() * 0.25 + 0.8; 
            action[FR_2] = out.index({5}).item().toFloat() * 0.25 - 1.5; 
            action[FL_0] = out.index({0}).item().toFloat() * 0.25 + 0.1; 
            action[FL_1] = out.index({1}).item().toFloat() * 0.25 + 0.8; 
            action[FL_2] = out.index({2}).item().toFloat() * 0.25 - 1.5; 
            action[RR_0] = out.index({9}).item().toFloat() * 0.25 - 0.1; 
            action[RR_1] = out.index({10}).item().toFloat() * 0.25 + 1.0; 
            action[RR_2] = out.index({11}).item().toFloat() * 0.25 - 1.5; 
            action[RL_0] = out.index({6}).item().toFloat() * 0.25 + 0.1; 
            action[RL_1] = out.index({7}).item().toFloat() * 0.25 + 1.0; 
            action[RL_2] = out.index({8}).item().toFloat() * 0.25 - 1.5; 
            // cout << "output action: "; 
            // for (i = 0; i < 12; i ++){
            //     cout << out.index({i}).item().toFloat() << ", "; 
            // }
            cout << endl; 
            last_action[FR_0] = out.index({3}).item().toFloat(); 
            last_action[FR_1] = out.index({4}).item().toFloat(); 
            last_action[FR_2] = out.index({5}).item().toFloat(); 
            last_action[FL_0] = out.index({0}).item().toFloat(); 
            last_action[FL_1] = out.index({1}).item().toFloat(); 
            last_action[FL_2] = out.index({2}).item().toFloat(); 
            last_action[RR_0] = out.index({9}).item().toFloat(); 
            last_action[RR_1] = out.index({10}).item().toFloat(); 
            last_action[RR_2] = out.index({11}).item().toFloat(); 
            last_action[RL_0] = out.index({6}).item().toFloat(); 
            last_action[RL_1] = out.index({7}).item().toFloat(); 
            last_action[RL_2] = out.index({8}).item().toFloat(); 
            for (i = 0; i < 12; i ++){
                if (last_action[i] > 100) last_action[i] = 100; 
                if (last_action[i] < -100) last_action[i] = -100; 
            }
        }
        
        // for (i = 0; i < 12; i++ ){
        //     if (motiontime >= 400){
        //         action[i] = jointLinearInterpolation(last_action[i], action[i], 0.2); 
        //     }
        // }
        // for (i = 0; i < 12; i ++){
        //     last_action[i] = action[i]; 
        // }
        // action = clipSafeTorque(action); 
        // cout << "Action: "; 
        // for (i = 0; i < 12; i ++){
        //     cout << action[sequence[i]] << ", ";  
        // }
        cout << endl; 
        for (i = 0; i < 12; i ++){
            cmd.motorCmd[i].q = action[i]; 
            cmd.motorCmd[i].dq = 0; 
            cmd.motorCmd[i].Kp = 20; 
            cmd.motorCmd[i].Kd = 0.5; 
        }
    }
    
    if(motiontime > 10){
        safe.PositionLimit(cmd);
        safe.PowerProtect(cmd, state, 5);
        // You can uncomment it for position protection
        // int res2 = safe.PositionProtect(cmd, state, 0.087);
        // if(res1 < 0) exit(-1);
    }
    end = clock(); 
    cout << "time cost: " << float(end - start) / CLOCKS_PER_SEC << endl; 

    udp.SetSend(cmd);

}


int main(void)
{

    double exp_vel_x, exp_vel_y, exp_vel_w; 
    cout << "Expected Velocity x: "; 
    cin >> exp_vel_x; 
    cout << "Expected Velocity y: ";
    cin >> exp_vel_y;
    cout << "Expected yaw velocity: ";
    cin >> exp_vel_w;

    std::cout << "Communication level is set to LOW-level." << std::endl
              << "WARNING: Make sure the robot is hung up." << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();
    
    Custom custom(LOWLEVEL, exp_vel_x, exp_vel_y, exp_vel_w);
    // InitEnvironment();
    LoopFunc loop_control("control_loop", custom.dt,    boost::bind(&Custom::RobotControl, &custom));
    LoopFunc loop_udpSend("udp_send",     custom.dt, 3, boost::bind(&Custom::UDPSend,      &custom));
    LoopFunc loop_udpRecv("udp_recv",     custom.dt, 3, boost::bind(&Custom::UDPRecv,      &custom));

    loop_udpSend.start();
    loop_udpRecv.start();
    loop_control.start();

    while(1){
        sleep(10);
    };

    return 0; 
}
