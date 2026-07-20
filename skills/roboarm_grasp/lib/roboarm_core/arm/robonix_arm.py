"""RobonixArm — 通过 ROS2 joint_command / joint_states 驱动 roboarm_arm 原语。"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from pathlib import Path

import kinpy
import numpy as np
from builtin_interfaces.msg import Time
from roboarm_core.arm.arm_base import Arm, StepCallback
from robonix_api.ros import RosBackend
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

GRIPPER_JOINT_NAME = "gripper"
JOINT_COUNT = 5


def _make_header(frame_id: str = "base_link") -> Header:
    now = time.time()
    sec = int(now)
    nanosec = int((now - sec) * 1e9)
    return Header(stamp=Time(sec=sec, nanosec=nanosec), frame_id=frame_id)


class RobonixArm(Arm):
    """通过 ROS2 话题与 roboarm_arm 原语通信的机械臂适配器。"""

    def __init__(
        self,
        *,
        joint_names: Sequence[str],
        arm_offset: Sequence[float],
        assets_root: Path | str,
        joint_command_topic: str,
        joint_states_topic: str,
        gripper_open_width_m: float = 0.080,
        motion_steps: int = 20,
        hand_eye_calibration_file: str | Path | None = None,
        urdf_path: str | Path | None = None,
    ) -> None:
        assets_root = Path(assets_root)
        if hand_eye_calibration_file is None:
            hand_eye_calibration_file = assets_root / "hand_eye" / "2d_homography.npy"
        if urdf_path is None:
            urdf_path = assets_root / "urdf" / "lerobo" / "low_cost_robot.urdf"
        super().__init__(hand_eye_calibration_file=hand_eye_calibration_file)

        self._joint_names = list(joint_names)
        self._arm_joint_names = [
            name for name in self._joint_names if name != GRIPPER_JOINT_NAME
        ]
        if len(self._arm_joint_names) != JOINT_COUNT:
            raise ValueError(
                f"joint_names must contain {JOINT_COUNT} arm joints plus gripper"
            )

        self._offset = [float(value) for value in arm_offset]
        if len(self._offset) != JOINT_COUNT:
            raise ValueError(f"arm_offset must contain {JOINT_COUNT} values")

        self._joint_command_topic = joint_command_topic
        self._joint_states_topic = joint_states_topic
        self._gripper_open_width_m = float(gripper_open_width_m)
        self._motion_steps = int(motion_steps)

        self._latest_joint_state: JointState | None = None
        self._state_lock = threading.Lock()
        self._cmd_pub = None
        self._state_sub = None
        self._connected = False

        with open(urdf_path, encoding="utf-8") as handle:
            self.chain = kinpy.build_serial_chain_from_urdf(
                handle.read(), "gripper_static_1"
            )

    def connect(self) -> None:
        if self._connected:
            return
        ros = RosBackend.get()

        def _on_joint_state(msg: JointState) -> None:
            with self._state_lock:
                self._latest_joint_state = msg

        self._state_sub = ros.create_subscription(
            JointState,
            self._joint_states_topic,
            _on_joint_state,
            "reliable",
        )
        if not ros.wait_for_topic(
            self._joint_states_topic, JointState, timeout_s=30.0
        ):
            raise TimeoutError(
                f"Timed out waiting for joint_states on {self._joint_states_topic}"
            )
        self._cmd_pub = ros.create_publisher(
            JointState,
            self._joint_command_topic,
            "reliable",
        )
        self._connected = True

    def disconnect_arm(self) -> None:
        self._cmd_pub = None
        self._state_sub = None
        with self._state_lock:
            self._latest_joint_state = None
        self._connected = False

    def get_raw_joint_angles(
        self, retry_times=None
    ) -> tuple[list[float] | None, float | None]:
        angles_deg, gripper_open_0to1 = self._read_joint_state()
        if angles_deg is None or gripper_open_0to1 is None:
            if retry_times is None:
                retry_times = self.get_arm_angles_retry_times
            if retry_times > 0:
                time.sleep(self.catch_time_interval_s)
                return self.get_raw_joint_angles(retry_times - 1)
            return None, None
        raw_angles = [
            angle + offset
            for angle, offset in zip(angles_deg, self._offset, strict=True)
        ]
        return raw_angles, gripper_open_0to1

    def get_arm_angles(
        self, retry_times=None
    ) -> tuple[list[float] | None, float | None]:
        angles_deg, gripper_open_0to1 = self._read_joint_state()
        if angles_deg is None or gripper_open_0to1 is None:
            if retry_times is None:
                retry_times = self.get_arm_angles_retry_times
            if retry_times > 0:
                time.sleep(self.catch_time_interval_s)
                return self.get_arm_angles(retry_times - 1)
            return None, None
        return angles_deg, gripper_open_0to1

    def set_arm_angles(
        self,
        angles_deg: Sequence[float | int] | None = None,
        gripper_open_0to1: float | None = None,
        step_callback: StepCallback | None = None,
        *,
        block_until_reach: bool = False,
    ) -> bool:
        if not self._connected or self._cmd_pub is None:
            return False
        target_joint_angles = None if angles_deg is None else list(angles_deg)
        target_gripper = gripper_open_0to1
        if target_gripper is not None and not 0 <= target_gripper <= 1:
            raise ValueError("gripper_open_0to1 must in [0, 1]")
        if target_joint_angles is None and target_gripper is None:
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
            current_gripper_0to1 if target_gripper is None else float(target_gripper)
        )
        current_joint_angles = list(current_angles_deg)
        for step_index, alpha in enumerate(
            np.linspace(0, 1, self._motion_steps + 1)[1:]
        ):
            interp_joint_angles = [
                current + (desired - current) * float(alpha)
                for current, desired in zip(
                    current_joint_angles, desired_joint_angles, strict=True
                )
            ]
            interp_gripper = current_gripper_0to1 + (
                desired_gripper - current_gripper_0to1
            ) * float(alpha)
            self._publish_joint_command(interp_joint_angles, interp_gripper)
            if step_callback is not None:
                step_callback(
                    {
                        "target_joint_angles_deg": list(interp_joint_angles),
                        "target_gripper_open_0to1": float(interp_gripper),
                        "step_index": int(step_index),
                        "steps": int(self._motion_steps),
                        "alpha": float(alpha),
                    }
                )
            time.sleep(0.5 / self._motion_steps)

        if block_until_reach and target_joint_angles is not None:
            try:
                self.wait_until_reached(desired_joint_angles)
            except TimeoutError:
                return False
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
            block_until_reach=block_until_reach,
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
            print("Warning: euler_angles_deg_zyx not supported in RobonixArm")
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
            return False
        angles_deg = np.rad2deg(result.x).tolist()
        return self.set_arm_angles(
            angles_deg,
            gripper_open_0to1=gripper_open_0to1,
            step_callback=step_callback,
            block_until_reach=block_until_reach,
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

    def _read_joint_state(self) -> tuple[list[float] | None, float | None]:
        with self._state_lock:
            msg = self._latest_joint_state
        if msg is None or not msg.name or not msg.position:
            return None, None
        name_to_position = {
            name: float(position)
            for name, position in zip(msg.name, msg.position, strict=True)
        }
        arm_positions_rad: list[float] = []
        for joint_name in self._arm_joint_names:
            if joint_name not in name_to_position:
                return None, None
            arm_positions_rad.append(name_to_position[joint_name])

        gripper_open_0to1: float | None = None
        if GRIPPER_JOINT_NAME in name_to_position and self._gripper_open_width_m > 0.0:
            gripper_open_0to1 = (
                name_to_position[GRIPPER_JOINT_NAME] / self._gripper_open_width_m
            )
            gripper_open_0to1 = float(max(0.0, min(1.0, gripper_open_0to1)))
        if gripper_open_0to1 is None:
            return None, None
        angles_deg = [float(np.rad2deg(angle_rad)) for angle_rad in arm_positions_rad]
        return angles_deg, gripper_open_0to1

    def _publish_joint_command(
        self,
        angles_deg: Sequence[float],
        gripper_open_0to1: float,
    ) -> None:
        if self._cmd_pub is None:
            raise RuntimeError("RobonixArm is not connected")
        msg = JointState()
        msg.header = _make_header()
        msg.name = list(self._joint_names)
        positions: list[float] = [
            float(np.deg2rad(angle_deg)) for angle_deg in angles_deg
        ]
        positions.append(float(gripper_open_0to1 * self._gripper_open_width_m))
        msg.position = positions
        self._cmd_pub.publish(msg)
