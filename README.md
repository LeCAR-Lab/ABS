# ABS

Official Implementation for [Agile But Safe: Learning Collision-Free High-Speed Legged Locomotion](https://agile-but-safe.github.io/).

[Tairan He*](https://tairanhe.com/), [Chong Zhang*](https://zita-ch.github.io/), [Wenli Xiao](https://wenlixiao-cs.github.io/), [Guanqi He](https://guanqihe.github.io/), [Changliu Liu](https://www.cs.cmu.edu/~cliu6/), [Guanya Shi](https://www.gshi.me/)  

<p align="center">
  <img src="images/Youtube-Cover[2M].png" width="80%"/>
</p>


## Hardware Deployment 
### System overview
<p align="center">
  <img src="images/hardware.png" width="80%"/>
</p>

- **Robot**: [Unitree Go1 EDU](https://shop.unitree.com/)
- **Perception**: [ZED mini Camera](https://store.stereolabs.com/products/zed-mini)
- **Onboard Compute**: [Orin NX (16GB)](https://www.seeedstudio.com/reComputer-J4012-p-5586.html)
- **LED**: [PSEQT LED Lights](https://www.amazon.com/gp/product/B0BKGF3JMG/ref=ox_sc_act_title_1?smid=A1QWPB2EZDWX2O&th=1)
- **Power Regulator**: [Pololu 12V, 15A Step-Down Voltage Regulator D24V150F12](https://www.pololu.com/product/2885)

### 3D Print Mounts
- Orin NX mount: [STL-PCMount_v2](deployment/3dprints/PCMount_v2.STL)
- ZED mini mount: [STL-CameraSeat](deployment/3dprints/CameraSeat.STL) and [STL-ZEDMountv1](deployment/3dprints/ZEDMountv1.STL)

### Deployment Code Installation
- [Unitree Go1 SDK](https://github.com/unitreerobotics/unitree_legged_sdk)
- [ZED SDK](https://www.stereolabs.com/developers/release)
- [ROS Noetic](https://wiki.ros.org/noetic/Installation/Ubuntu)
- Pytorch on a Python 3 environment

### Deployment Setup
- Low Level Control Mode for Unitree Go1: `L2+A` -> `L2+B` -> `L1+L2+Start`
- Network Configuration for Orin NX: 
  - **IP**: `192.168.123.15`
  - **Netmask**: `255.255.255.0`
- Convert the `.pt` files of agile/recovery policy, RA value to `.onnx` files using `src/abs_src/onnx_model_converter.py`

- Modify the path of (.onnx or .pt) models in `publisher_depthimg_linvel.py` and `depth_obstacle_depth_goal_ros.py`

### Deployment Scripts
1. `roscore`: Activate ROS Noetic Envrioment
2. `cd src/abs_src/`: Enter the ABS scripts file
3. `python publisher_depthimg_linvel.py`: Publish ray prediction results and odometry results for navigation goals
4. `python led_control_ros.py`: Control the two LED lights based on RA values
5. `python depth_obstacle_depth_goal_ros.py`: Activate the Go1 using the agile policy and the recovery policy

### Deployment Controllers
- `B`: Emergence stop
- `Default`: Go1 Running ABS based on goal command
- `L2`: Turn left
- `R2`: Turn right
- `Down`: Back
- `Up`: Stand
- `A`: Turn around
- `X`: Back to initial position



## TODO:

- [ ] Upload Sim Training Code
- [x] Upload Deployment Code

