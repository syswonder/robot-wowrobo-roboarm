本库基于 https://github.com/AuYang261/roboarm 修改而来
标定脚本测试较少，如果有问题可以通过上面的库进行标定，然后把文件传入本库中

## 项目目录：
包含两个Primitives:
    - roboarm_arm: wowrobo品牌roboarm机械臂
    - orbbec_camera_roboarm: orbbec品牌RGB-D摄像头，型号Gemini 215 SN
包含一个Skill：
    - roboarm_grasp: 使用机械臂进行抓取的技能，包括classify and grasp和catch by instruction

## 注意事项：
你需要格外注意两个配置文件:
    ./robonix_manifest.yaml: 起robonix时传入的参数
    ./skills/roboarm_grasp/config/config.yaml: 抓取skill的一些特有配置参数
    ./tools/config.yaml: 各类小工具可能需要的配置参数

## 使用教程：
1.确保你使用的是python>=3.10，并已经装好了robonix，推荐使用conda进行环境管理

2.安装依赖
```bash
pip install -r ./requirements.txt
```

3.安装相机sdk
先克隆仓库 https://github.com/orbbec/pyorbbecsdk/tree/v2-main 
按 https://orbbec.github.io/pyorbbecsdk/source/2_installation/registration_script.html#quick-setup-recommended 注册元数据，以便你的电脑能解析相机数据。macos不用注册即可用。
```bash
cd pyorbbecsdk/scripts/env_setup
python setup_env.py
```
再安装pyorbbecsdk。下载.whl文件， https://github.com/orbbec/pyorbbecsdk/releases/tag/v2.0.13
在其所在目录下执行pip install pyorbbecsdk-2.0.13-cp310-cp310-win_amd64.whl安装。

4.根据 tools 文件夹中README所述测试设备连接，并进行标定

5.编译robonix
```bash
rbnx build
```

6.运行robonix
```bash
rbnx boot -f robonix_manifest.yaml
```

7.在另一bash上测试连接
```bash
rbnx caps -v
```
应该出现roboarm_arm,orbbec_camera_roboarm,roboarm_grasp等
并测试ROS2 topic
```bash
source /opt/ros/humble/setup.bash
source ~/roboarm/robonix/roboarm/primitives/roboarm_arm/rbnx-build/codegen/ros2_idl/install/setup.bash
ros2 topic list
```
应该看到/roboarm/joint_states`、`/roboarm/end_pose`、`/roboarm/joint_command`、`/roboarm/camera/rgb等话题

8.使用pilot执行操作
```bash
rbnx chat
```
在对话中输入
```
对当前画面中的积木进行分类检测并逐个抓取
```
测试classify and grasp
```
抓取红色积木
```
测试catch by instruction