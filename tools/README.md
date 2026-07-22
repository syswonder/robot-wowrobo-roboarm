# robonix/roboarm 离线工具

与 `primitives/`、`skills/` 同级，用于标定和调试。**不会**被 `rbnx boot` 加载，也**不需要** ROS2 / Robonix 运行时。

## 目录

```
tools/
├── config.yaml              # 工具专用配置（不影响 Robonix 运行时）
├── lib/
│   ├── bootstrap.py         # 初始化 PYTHONPATH（不依赖 Robonix / ROS）
│   ├── config.py            # 工具专用配置加载
│   ├── arm_base.py          # 机械臂控制基类
│   ├── yolo_detect.py       # YOLO 检测
│   ├── paths.py             # 指向 primitive/skill 内 assets
│   ├── lerobo_arm.py        # 直连 LeRobot 臂（标定/调试）
│   ├── local_camera.py      # 直连 Orbbec 相机
│   └── cv2_display.py       # OpenCV 窗口 / 无头 Web 预览
├── calibrate_offset.py      # 关节 offset 标定
├── orb_camera.py            # 测试相机画面
├── calibrate_handeye_2d.py  # 2D 手眼标定
├── control_by_pos.py        # 键盘控制末端位置
├── classify_and_grasp.py    # 离线 YOLO 分类抓取（调试参数）
└── detect_yolo.py           # YOLO 检测预览
```

## 依赖

```bash
# 在根目录下安装过依赖便无需这步
cd ~/roboarm/robonix/roboarm/tools
pip install -r requirements.txt
# lerobot 已由 primitives/roboarm_arm/src/vendor/lerobot 提供
```

## 配置

编辑 `tools/config.yaml`：

- `arm_port`：你的机械臂串口（如 `/dev/ttyACM0`）
- 其余根据需要修改

## 用法

均在 `tools/` 目录下运行。**运行工具前请先停止 `rbnx boot`**，否则串口会被 Robonix 原语占用。

```bash
cd ~/roboarm/robonix/roboarm/tools
```

**0. 测试相机正常工作**
```bash
python orb_camera.py
```
若能正常弹出显示图像（彩色图和深度图）的窗口，说明相机配置正常。
如果失败，需要降低numpy版本到<2

**1. 标定关节 offset：**
```bash
python calibrate_offset.py
```
进入标定程序后，将机械臂摆放为初始状态（大致样子即可，后一步会精确修正），按回车，进入下一步。
接下来会不断显示各关节角度值，并会随机械臂移动而变化。此时转动机械臂各个关节，在期望的工作范围内转动即可，不需要转过头。都转完后按回车，进入下一步。
如果转过头或者连接不好会断联，重新标定即可
接下来会不断输出当前各关节角度值（以=======为分割）。此时将机械臂摆放到初始状态（需要比较精确地摆放）：
按ctrl+c键停止程序，最后输出的一组角度值（以=======为分割，5个为一组）作为5个关节的offset。将这5个offset复制到配置文件robonix_manifest.yaml中的arm_offset项下：

**2. 测试机械臂控制**
```bash
python control_by_pos.py
```
刚启动时机械臂应该在初始状态，通过键盘控制机械臂移动，
w/s控制前后，a/d控制左右，空格/shift控制夹爪升降，z/c控制夹爪旋转，q/e控制夹爪开闭，按Esc退出。
若全部正常说明正确完成了标定。

**3. 手眼标定（输出到 skills/roboarm_grasp/assets/hand_eye/2d_homography.npy）**
```bash
python calibrate_handeye_2d.py --mode calibrate
```
- 准备一个显眼的点（如一个小螺丝），用鼠标点击图片上该点的位置
- 再将机械臂末端移动到该位置，要求夹爪static连杆垂直于桌面，即保证夹爪根部和末端xy坐标相同，按空格键记录机械臂末端位姿
- 这样鼠标点一次和按一次空格键就是一组数据。改变点的位置，重复收集4组以上数据，数据越多误差越小，一般10组即可。收集完毕后按ESC键退出计算标定结果。不能点错，否则要重新运行上述命令重新进行标定。
- 标定的结果是得到一个矩阵（自动保存），表示相机坐标系（二维）到机械臂基座坐标系（z轴为桌面不变，故也是二维）的变换，从而可以完成图片上的像素坐标系到机械臂坐标系的转换

**4.测试标定结果：**
```bash
python calibrate_handeye_2d.py --mode test
```
在弹出的相机画面窗口中点击想要移动到的点，机械臂应当会移动到这一点，说明标定正常。

**5.安装YOLO模型**
下载训练好的YOLO模型（以.pt结尾），放在 skills\roboarm_grasp\assets\models\yolo\积木方块 目录下，
将配置文件config.yaml中的classification_YOLO_model_path改为模型保存的路径(如果你要换用别的路径或模型)。
用于识别积木方块的模型如下：
https://www.modelscope.cn/models/AuYang03/YOLO-block-detect/file/view/master/best.pt?status=2

**6. YOLO 实时检测预览**
```bash
python detect_yolo.py
```

**7. 离线 classify and grasp（调试抓取参数，不启动 Robonix）**
```bash
python classify_and_grasp.py
```

- 实时显示 YOLO 检测框
- **g** 或 **空格**：对当前画面检测结果执行抓取（逻辑与 skill `classify_and_grasp` 一致）
- **h**：回 home
- **Esc**：退出

抓取/放置参数（`place_pos`、`catch_raise_height` 等）默认读取
`skills/roboarm_grasp/config/config.yaml`（由 `tools/config.yaml` 的 `grasp_config` 指定）。
修改 skill 配置后重启本工具即可调试，无需改 Robonix manifest。

单轮抓取（等同 MCP 一次调用）：
```bash
python classify_and_grasp.py --once
```

使用 skill 配置覆盖全部参数（仍保留 tools 里的 `arm_port` 等）：
```bash
TOOLS_CONFIG=config.yaml python classify_and_grasp.py
```
