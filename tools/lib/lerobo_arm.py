from __future__ import annotations

import os
import time
from collections.abc import Sequence
from pathlib import Path

import kinpy
import numpy as np
from lib.arm_base import Arm, StepCallback
from lib.config import get_config_value
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from lib.paths import calibration_dir, hand_eye_matrix_path, urdf_path

from lerobot.robots.koch_follower import config_koch_follower, koch_follower


class LeroboArm(Arm):
    """Direct LeRobot Koch follower control for offline calibration tools."""

    MAX_GRIPPER_ANGLE_DEG = 100

    def __init__(
        self,
        *,
        robot_id: str = "koch_follower",
        hand_eye_calibration_file: str | Path | None = None,
    ) -> None:
        if hand_eye_calibration_file is None:
            hand_eye_calibration_file = hand_eye_matrix_path()
        super().__init__(hand_eye_calibration_file=hand_eye_calibration_file)

        self.arm_backend = "real"
        self.steps = int(get_config_value("motion_steps", 20, raise_if_missing=False))
        self.offset = [float(v) for v in get_config_value("arm_offset")]
        if len(self.offset) != 5:
            raise ValueError(
                "config.yaml 中 arm_offset 必须是 5 个关节角度；"
                "标定 offset 时可临时设为 [0,0,0,0,0]"
            )

        with open(urdf_path(), encoding="utf-8") as handle:
            self.chain = kinpy.build_serial_chain_from_urdf(
                handle.read(), "gripper_static_1"
            )

        port = str(get_config_value("arm_port"))
        self.arm = koch_follower.KochFollower(
            config_koch_follower.KochFollowerConfig(
                port=port,
                disable_torque_on_disconnect=True,
                use_degrees=True,
                id=robot_id,
                calibration_dir=Path(calibration_dir()).resolve(),
            )
        )
        try:
            self.arm.connect()
        except ConnectionError as exc:
            raise ConnectionError(
                f"机械臂连接失败: {exc}\n"
                f"当前 arm_port={port}\n"
                "排查步骤:\n"
                "  1. 先停止 rbnx boot（串口不能同时被占用）\n"
                "  2. 运行 python check_arm_port.py 扫描可用串口\n"
                "  3. 确认 USB 已接入 WSL: ls -l /dev/ttyACM*\n"
                "  4. 权限: sudo chmod 666 /dev/ttyACM0 或 sudo usermod -aG dialout $USER\n"
                "  5. 修改 tools/config.yaml 中的 arm_port"
            ) from exc

    def _joint_names(self) -> list[str]:
        return list(self.arm.bus.motors.keys())

    def get_raw_joint_angles(
        self, retry_times=None
    ) -> tuple[list[float] | None, float | None]:
        try:
            angles_deg = list(self.arm.get_observation().values())
            gripper = float(
                np.clip(angles_deg[-1] / self.MAX_GRIPPER_ANGLE_DEG, 0.0, 1.0)
            )
            return angles_deg[:-1], gripper
        except Exception:
            if retry_times is None:
                retry_times = self.get_arm_angles_retry_times
            if retry_times > 0:
                time.sleep(self.catch_time_interval_s)
                return self.get_raw_joint_angles(retry_times - 1)
            return None, None

    def get_arm_angles(
        self, retry_times=None
    ) -> tuple[list[float] | None, float | None]:
        raw_angles_deg, gripper_open_0to1 = self.get_raw_joint_angles(retry_times)
        if raw_angles_deg is None or gripper_open_0to1 is None:
            return None, None
        return [
            angle - offset
            for angle, offset in zip(raw_angles_deg, self.offset, strict=True)
        ], gripper_open_0to1

    def set_arm_angles(
        self,
        angles_deg: Sequence[float | int] | None = None,
        gripper_open_0to1: float | None = None,
        step_callback: StepCallback | None = None,
        *,
        block_until_reach: bool = False,
    ) -> bool:
        joint_names = self._joint_names()
        target_joint_angles = None if angles_deg is None else list(angles_deg)
        if gripper_open_0to1 is not None and not 0 <= gripper_open_0to1 <= 1:
            raise ValueError("gripper_open_0to1 must in [0, 1]")
        if target_joint_angles is None and gripper_open_0to1 is None:
            return True

        current_angles_deg, current_gripper_0to1 = self.get_arm_angles()
        if current_angles_deg is None or current_gripper_0to1 is None:
            return False

        desired_joint_angles = list(current_angles_deg)
        if target_joint_angles is not None:
            for index, angle_deg in enumerate(target_joint_angles):
                if angle_deg is not None:
                    desired_joint_angles[index] = float(
                        np.clip(angle_deg, -180, 180)
                    )
        desired_gripper = (
            current_gripper_0to1
            if gripper_open_0to1 is None
            else float(gripper_open_0to1)
        )

        current_joint_angles = list(current_angles_deg)
        current_gripper_angle_deg = current_gripper_0to1 * self.MAX_GRIPPER_ANGLE_DEG
        desired_gripper_angle_deg = desired_gripper * self.MAX_GRIPPER_ANGLE_DEG

        for step_index, alpha in enumerate(np.linspace(0, 1, self.steps + 1)[1:]):
            interp_joint_angles = [
                current + (desired - current) * float(alpha)
                for current, desired in zip(
                    current_joint_angles, desired_joint_angles, strict=True
                )
            ]
            interp_gripper_angle_deg = (
                current_gripper_angle_deg * (1 - alpha)
                + desired_gripper_angle_deg * alpha
            )
            action = {
                motor_name + ".pos": interp_angle + self.offset[index]
                for index, (motor_name, interp_angle) in enumerate(
                    zip(joint_names[:-1], interp_joint_angles, strict=True)
                )
            }
            action["gripper.pos"] = interp_gripper_angle_deg
            try:
                self.arm.send_action(action)
            except Exception as exc:
                print(f"设置机械臂角度失败: {exc}")
                return False
            if step_callback is not None:
                step_callback(
                    {
                        "step_index": int(step_index),
                        "steps": int(self.steps),
                        "alpha": float(alpha),
                    }
                )
            time.sleep(0.5 / self.steps)
        return True

    def get_arm_pose(self) -> tuple[list[float] | None, list[float] | None]:
        angles_deg, _ = self.get_arm_angles()
        if angles_deg is None:
            return None, None
        fk: kinpy.Transform = self.chain.forward_kinematics(
            np.deg2rad(angles_deg).tolist()
        )  # type: ignore[assignment]
        return (
            fk.pos.tolist(),
            R.from_euler("xyz", fk.rot_euler, degrees=True)
            .as_euler("zyx", degrees=True)
            .tolist(),
        )

    def move_to_home(
        self,
        gripper_open_0to1: float | int | None = None,
        step_callback: StepCallback | None = None,
        *,
        block_until_reach: bool = False,
    ) -> bool:
        return self.set_arm_angles(
            [0, 0, 0, 0, 0],
            gripper_open_0to1=gripper_open_0to1,
            step_callback=step_callback,
        )

    def move_to(
        self,
        pos: list[float],
        gripper_open_0to1: float | int | None = None,
        rot_rad: float | int | None = None,
        euler_angles_deg_zyx: list[float] | None = None,
        step_callback: StepCallback | None = None,
        *,
        block_until_reach: bool = False,
    ) -> bool:
        if euler_angles_deg_zyx:
            print("Warning: euler_angles_deg_zyx not supported in LeroboArm tools")
        if len(pos) != 3:
            raise ValueError("位置参数格式错误，应该是[x, y, z]")
        goal_tf = kinpy.Transform(
            pos=np.array(pos), rot=[0, 0, -rot_rad if rot_rad else 0]
        )
        result = minimize(
            self._ik_cost_function,
            x0=np.zeros(len(self.chain.get_joint_parameter_names())),
            args=(goal_tf.matrix(), self.chain),
            method="SLSQP",
        )
        if not result.success:
            print("逆运动学不收敛，无法到达指定位置")
            return False
        return self.set_arm_angles(
            np.rad2deg(result.x).tolist(),
            gripper_open_0to1=gripper_open_0to1,
            step_callback=step_callback,
        )

    def set_gripper(
        self,
        gripper_open_0to1: float,
        step_callback: StepCallback | None = None,
    ) -> bool:
        return self.set_arm_angles(
            gripper_open_0to1=gripper_open_0to1,
            step_callback=step_callback,
        )

    def disconnect_arm(self) -> None:
        self.arm.disconnect()

    def enable_torque(self) -> None:
        self.arm.bus.enable_torque()

    def disable_torque(self) -> None:
        self.arm.bus.disable_torque()
