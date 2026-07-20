from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from math import inf
from typing import Any

import cv2
import numpy as np
from lib.config import get_config_value
from scipy.spatial.transform import Rotation as R

StepCallback = Callable[[dict[str, Any]], None]


class Arm:
    """机械臂控制基类：抓取、放置与手眼坐标转换。"""

    IK_POSITION_TOLERANCE_M = 0.01
    IK_TILT_TOLERANCE_RAD = np.deg2rad(5.0)
    IK_YAW_TOLERANCE_RAD = np.deg2rad(10.0)

    def __init__(self, hand_eye_calibration_file: str | os.PathLike[str]) -> None:
        self.timeout_s = 3
        self.desktop_height = get_config_value("default_desktop_height")
        self.catch_raise_height = get_config_value("catch_raise_height", 0.1)
        self.place_raise_height = get_config_value("place_raise_height", 0.1)
        self.default_gripper_close_threshold = get_config_value(
            "default_gripper_close_threshold"
        )
        self.catch_time_interval_s = get_config_value("catch_time_interval_s")
        self.get_arm_angles_retry_times = get_config_value("get_arm_angles_retry_times")
        self.reach_mse_threshold = float(
            get_config_value("arm_reach_mse_threshold_deg2")
        )
        if os.path.exists(hand_eye_calibration_file):
            self.hand_eye_calibration_matrix = np.load(hand_eye_calibration_file)

    def set_arm_angles(
        self,
        angles_deg: Sequence[float | int] | None = None,
        gripper_open_0to1: float | None = None,
        step_callback: StepCallback | None = None,
        *,
        block_until_reach: bool = False,
    ) -> bool:
        raise NotImplementedError

    def get_arm_angles(
        self, retry_times=None
    ) -> tuple[list[float] | None, float | None]:
        raise NotImplementedError

    def get_raw_joint_angles(
        self, retry_times=None
    ) -> tuple[list[float] | None, float | None]:
        raise NotImplementedError

    def get_arm_pose(self) -> tuple[list[float] | None, list[float] | None]:
        raise NotImplementedError

    def move_to_home(
        self,
        gripper_open_0to1: float | int | None = None,
        step_callback: StepCallback | None = None,
        *,
        block_until_reach: bool = False,
    ) -> bool:
        raise NotImplementedError

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
        raise NotImplementedError

    def wait_until_reached(self, target_angles_deg: Sequence[float]) -> None:
        target = [float(angle) for angle in target_angles_deg]
        deadline = time.monotonic() + self.timeout_s
        while True:
            current_angles_deg, _ = self.get_raw_joint_angles()
            if current_angles_deg is None:
                continue
            if len(current_angles_deg) != len(target):
                raise RuntimeError(
                    f"关节数不一致: {len(current_angles_deg)} vs {len(target)}"
                )
            squared_errors = [
                (current - desired) ** 2
                for current, desired in zip(current_angles_deg, target, strict=True)
            ]
            mse = sum(squared_errors) / len(squared_errors)
            if mse <= self.reach_mse_threshold:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"等待机械臂到位超时: MSE={mse:.4f} deg^2, "
                    f"阈值={self.reach_mse_threshold} deg^2"
                )
            time.sleep(0.1)

    def set_gripper(
        self,
        gripper_open_0to1: float,
        step_callback: StepCallback | None = None,
    ) -> bool:
        raise NotImplementedError

    def disconnect_arm(self) -> None:
        raise NotImplementedError

    def enable_torque(self) -> None:
        return

    def disable_torque(self) -> None:
        return

    @staticmethod
    def _wrap_angle_rad(angle_rad: float) -> float:
        return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))

    @classmethod
    def _ik_cost_function(cls, joint_angles, target_pose_matrix, chain):
        current_pose_matrix = chain.forward_kinematics(joint_angles).matrix()
        pos_error = np.linalg.norm(
            current_pose_matrix[:3, 3] - target_pose_matrix[:3, 3]
        )
        current_rot = R.from_matrix(current_pose_matrix[:3, :3])
        target_rot = R.from_matrix(target_pose_matrix[:3, :3])
        current_z_axis = current_rot.apply([0.0, 0.0, 1.0])
        target_z_axis = target_rot.apply([0.0, 0.0, 1.0])
        cosine = np.clip(np.dot(current_z_axis, target_z_axis), -1.0, 1.0)
        tilt_error = float(np.arccos(cosine))
        current_yaw = current_rot.as_euler("zyx")[0]
        target_yaw = target_rot.as_euler("zyx")[0]
        yaw_error = abs(cls._wrap_angle_rad(current_yaw - target_yaw))
        pos_term = (pos_error / cls.IK_POSITION_TOLERANCE_M) ** 2
        tilt_term = (tilt_error / cls.IK_TILT_TOLERANCE_RAD) ** 2
        yaw_term = (yaw_error / cls.IK_YAW_TOLERANCE_RAD) ** 2
        return float(pos_term + tilt_term + yaw_term)

    def catch(
        self,
        target_x: float,
        target_y: float,
        rot_rad: float,
        height: float = inf,
        step_callback: StepCallback | None = None,
    ) -> bool:
        target_z = self.desktop_height if height == inf else height
        res = self.move_to(
            [target_x, target_y, target_z + self.catch_raise_height],
            gripper_open_0to1=1,
            rot_rad=rot_rad,
            step_callback=step_callback,
        )
        if not res:
            self.move_to_home(gripper_open_0to1=1)
            return False
        time.sleep(self.catch_time_interval_s * 2)
        res = self.move_to(
            [target_x, target_y, target_z],
            gripper_open_0to1=1,
            rot_rad=rot_rad,
            step_callback=step_callback,
        )
        if not res:
            self.move_to_home(gripper_open_0to1=1)
            return False
        time.sleep(self.catch_time_interval_s)
        self.set_gripper(gripper_open_0to1=0, step_callback=step_callback)
        time.sleep(self.catch_time_interval_s)
        res = self.move_to(
            [target_x, target_y, target_z + self.catch_raise_height],
            rot_rad=rot_rad,
            step_callback=step_callback,
        )
        if not res:
            self.move_to_home(gripper_open_0to1=1)
            return False
        time.sleep(self.catch_time_interval_s)
        _, current_gripper_open_0to1 = self.get_arm_angles()
        if (
            current_gripper_open_0to1 is None
            or current_gripper_open_0to1 < self.default_gripper_close_threshold
        ):
            self.move_to_home(gripper_open_0to1=1)
            return False
        return True

    def place(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        rot_rad: float = 0,
        step_callback: StepCallback | None = None,
    ) -> bool:
        res = self.move_to(
            [target_x, target_y, target_z + self.place_raise_height],
            rot_rad=rot_rad,
            step_callback=step_callback,
        )
        if not res:
            self.move_to_home(gripper_open_0to1=1)
            return False
        time.sleep(self.catch_time_interval_s * 2)
        down = get_config_value(
            "go_down_before_open_gripper_in_place", False, raise_if_missing=False
        )
        if down:
            res = self.move_to(
                [target_x, target_y, target_z],
                rot_rad=rot_rad,
                step_callback=step_callback,
            )
            if not res:
                self.move_to_home(gripper_open_0to1=1)
                return False
            time.sleep(self.catch_time_interval_s)
        self.set_gripper(gripper_open_0to1=1, step_callback=step_callback)
        if down:
            res = self.move_to(
                [target_x, target_y, target_z + self.place_raise_height],
                rot_rad=rot_rad,
                step_callback=step_callback,
            )
            if not res:
                self.move_to_home(gripper_open_0to1=1)
                return False
            time.sleep(self.catch_time_interval_s)
        return True

    def catch_and_place(
        self,
        target_x: float,
        target_y: float,
        catch_rotate_rad: float,
        place_pos: list[float | int],
        height: float = inf,
        place_rotate_rad: float = 0,
        step_callback: StepCallback | None = None,
    ) -> bool:
        if len(place_pos) == 2:
            place_x, place_y = place_pos
            place_z = self.desktop_height
        elif len(place_pos) == 3:
            place_x, place_y, place_z = place_pos
        else:
            return False
        if not self.catch(
            target_x,
            target_y,
            catch_rotate_rad,
            height=height,
            step_callback=step_callback,
        ):
            self.move_to_home(gripper_open_0to1=1)
            return False
        if not self.place(
            place_x,
            place_y,
            place_z,
            place_rotate_rad,
            step_callback=step_callback,
        ):
            self.move_to_home(gripper_open_0to1=1)
            return False
        self.move_to_home(gripper_open_0to1=1, step_callback=step_callback)
        return True

    def pixel2pos(self, u: float, v: float) -> tuple[float, float]:
        if not hasattr(self, "hand_eye_calibration_matrix"):
            raise ValueError("没有手眼标定数据，无法转换图像坐标")
        pixel_coords = np.array([[u], [v], [1]])
        world_coords = self.hand_eye_calibration_matrix @ pixel_coords
        world_coords /= world_coords[2]
        return float(world_coords[0, 0]), float(world_coords[1, 0])

    @staticmethod
    def gripper_angle_by_longer(
        u: float, v: float, w: float, h: float, angle_deg: float
    ) -> float:
        box_points = cv2.boxPoints(((u, v), (w, h), angle_deg))
        if np.linalg.norm(box_points[0] - box_points[1]) > np.linalg.norm(
            box_points[1] - box_points[2]
        ):
            long_edge_points = (
                [box_points[0], box_points[1]]
                if box_points[0][0] < box_points[1][0]
                else [box_points[1], box_points[0]]
            )
        else:
            long_edge_points = (
                [box_points[1], box_points[2]]
                if box_points[1][0] < box_points[2][0]
                else [box_points[2], box_points[1]]
            )
        gripper_rot_rad = np.pi / 2 + np.arctan2(
            long_edge_points[1][1] - long_edge_points[0][1],
            long_edge_points[1][0] - long_edge_points[0][0],
        )
        if gripper_rot_rad > np.pi / 2:
            gripper_rot_rad -= np.pi
        return gripper_rot_rad
