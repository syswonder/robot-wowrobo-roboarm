#!/usr/bin/env python3
"""roboarm_arm — Robonix primitive for the LeRobot Koch follower arm."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import kinpy
import numpy as np
import requests
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from robonix_api import Err, Ok, Primitive
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

log = logging.getLogger("roboarm_arm")

JOINT_COUNT = 5
MAX_GRIPPER_ANGLE_DEG = 100.0
HOME_ANGLES_DEG = [0.0, 0.0, 0.0, 0.0, 0.0]
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

DEFAULT_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "gripper",
]
GRIPPER_JOINT_NAME = "gripper"


def _package_root(cfg: dict) -> Path:
    raw = cfg.get("package_root") or os.environ.get("RBNX_PACKAGE_ROOT")
    if raw:
        path = Path(str(raw)).resolve()
        if not path.is_dir():
            raise ValueError(f"package_root does not exist: {path}")
        return path
    return Path(__file__).resolve().parent.parent


def _ensure_lerobot_path(lerobot_src: Path) -> None:
    text = str(lerobot_src)
    if text not in sys.path:
        sys.path.insert(0, text)


class _SimArmClient:
    """Minimal HTTP client for Lerobo sim backend (Isaac Sim service)."""

    def __init__(self, host: str, port: int, timeout_s: float = 10.0) -> None:
        self._base_url = f"http://{host}:{port}"
        self._timeout_s = timeout_s

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        try:
            resp = requests.request(
                method,
                f"{self._base_url}{path}",
                json=payload,
                timeout=self._timeout_s,
            )
        except requests.RequestException as exc:
            raise ConnectionError(f"simulation service is unavailable: {exc}") from exc
        if not resp.ok:
            raise ConnectionError(
                f"simulation service request failed: {resp.status_code}; {resp.text}"
            )
        if not resp.text:
            return {}
        result = resp.json()
        if isinstance(result, dict) and result.get("ok") is False:
            raise RuntimeError(result.get("error") or "simulation service returned failure")
        return result

    def ping(self) -> dict:
        return self._request("GET", "/health")

    def get_raw_joint_angles(self) -> tuple[list[float], float]:
        result = self._request("GET", "/state")
        joint_angles = result.get("joint_angles_deg")
        gripper_open_0to1 = result.get("gripper_open_0to1")
        if not isinstance(joint_angles, list):
            raise RuntimeError("simulation service did not return joint_angles_deg")
        if not isinstance(gripper_open_0to1, (int, float)):
            raise RuntimeError("simulation service did not return gripper_open_0to1")
        return [float(angle) for angle in joint_angles], float(gripper_open_0to1)

    def send_joint_targets(
        self,
        joint_names: Sequence[str],
        joint_angles_deg: Sequence[float],
        gripper_open_0to1: float,
    ) -> None:
        self._request(
            "POST",
            "/state",
            {
                "joint_names": list(joint_names),
                "joint_angles_deg": [float(angle) for angle in joint_angles_deg],
                "gripper_open_0to1": float(gripper_open_0to1),
            },
        )

    def disconnect(self) -> None:
        return


class LeroboDriver:
    """Thread-safe wrapper around LeRobot KochFollower or the sim HTTP backend."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._lock = threading.RLock()
        self._arm = None
        self._sim: _SimArmClient | None = None
        self._chain: kinpy.serial_chain.SerialChain | None = None
        self._joint_names: list[str] = list(cfg["joint_names"])
        self._arm_joint_names = [
            name for name in self._joint_names if name != GRIPPER_JOINT_NAME
        ]
        self._offset = [float(value) for value in cfg["arm_offset"]]
        if len(self._offset) != JOINT_COUNT:
            raise ValueError(f"arm_offset must contain {JOINT_COUNT} values")

    @property
    def backend(self) -> str:
        return str(self._cfg.get("arm_backend", "real"))

    def connect(self) -> None:
        with self._lock:
            if self._arm is not None or self._sim is not None:
                return

            lerobot_src = Path(self._cfg["lerobot_src"])
            _ensure_lerobot_path(lerobot_src)

            urdf_path = Path(self._cfg["urdf_path"])
            with open(urdf_path, encoding="utf-8") as handle:
                self._chain = kinpy.build_serial_chain_from_urdf(
                    handle.read(), "gripper_static_1"
                )

            if self.backend == "sim":
                self._sim = _SimArmClient(
                    host=str(self._cfg["arm_sim_host"]),
                    port=int(self._cfg["arm_sim_port"]),
                    timeout_s=float(self._cfg.get("arm_sim_timeout_s", 10.0)),
                )
                deadline = time.monotonic() + float(self._cfg.get("connect_timeout_s", 30.0))
                while time.monotonic() < deadline:
                    try:
                        self._sim.ping()
                        break
                    except ConnectionError:
                        time.sleep(1.0)
                else:
                    raise ConnectionError("Timed out connecting to Lerobo simulation service")
                log.info(
                    "Lerobo sim connected at %s:%s",
                    self._cfg["arm_sim_host"],
                    self._cfg["arm_sim_port"],
                )
                return

            from lerobot.robots.koch_follower import config_koch_follower, koch_follower

            calibration_dir = Path(self._cfg["calibration_dir"]).resolve()
            self._arm = koch_follower.KochFollower(
                config_koch_follower.KochFollowerConfig(
                    port=str(self._cfg["serial_port"]),
                    disable_torque_on_disconnect=True,
                    use_degrees=True,
                    id=str(self._cfg.get("robot_id", "koch_follower")),
                    calibration_dir=calibration_dir,
                )
            )
            try:
                self._arm.connect()
            except ConnectionError as exc:
                raise ConnectionError(
                    f"Failed to connect Lerobo arm on {self._cfg['serial_port']}: {exc}"
                ) from exc
            log.info("Lerobo arm connected on %s", self._cfg["serial_port"])

    def disconnect(self) -> None:
        with self._lock:
            if self._arm is not None:
                try:
                    self._arm.disconnect()
                except Exception as exc:
                    log.warning("Error while disconnecting Lerobo arm: %s", exc)
                self._arm = None
            if self._sim is not None:
                self._sim.disconnect()
                self._sim = None
            log.info("Lerobo arm disconnected")

    def read_joint_state(
        self,
    ) -> tuple[list[float], float, list[float], np.ndarray]:
        angles_deg, gripper_open_0to1 = self._read_corrected_angles()
        chain = self._require_chain()
        fk: kinpy.Transform = chain.forward_kinematics(np.deg2rad(angles_deg).tolist())  # type: ignore[assignment]
        quat_xyzw = R.from_euler("xyz", fk.rot_euler, degrees=False).as_quat()
        positions_rad = [float(np.deg2rad(angle)) for angle in angles_deg]
        return positions_rad, gripper_open_0to1, fk.pos.tolist(), quat_xyzw

    def write_joint_command(
        self,
        joint_positions_rad: Sequence[float],
        *,
        gripper_open_0to1: float | None = None,
    ) -> None:
        if len(joint_positions_rad) != JOINT_COUNT:
            raise ValueError(
                f"Expected {JOINT_COUNT} joint positions, got {len(joint_positions_rad)}"
            )
        angles_deg = [
            float(np.clip(np.rad2deg(angle_rad), -180.0, 180.0))
            for angle_rad in joint_positions_rad
        ]
        self._send_joint_targets(
            angles_deg,
            gripper_open_0to1=gripper_open_0to1,
            interpolate=False,
        )

    def move_to_home(self, *, gripper_open_0to1: float | None = 1.0) -> None:
        self._send_joint_targets(
            HOME_ANGLES_DEG,
            gripper_open_0to1=gripper_open_0to1,
            interpolate=True,
        )

    def _require_chain(self) -> kinpy.serial_chain.SerialChain:
        if self._chain is None:
            raise RuntimeError("Lerobo kinematics chain is not initialized")
        return self._chain

    def _read_raw_angles(self) -> tuple[list[float], float]:
        with self._lock:
            if self._sim is not None:
                return self._sim.get_raw_joint_angles()
            if self._arm is None:
                raise RuntimeError("Lerobo arm is not connected")
            values = list(self._arm.get_observation().values())
            gripper_open_0to1 = float(
                np.clip(values[-1] / MAX_GRIPPER_ANGLE_DEG, 0.0, 1.0)
            )
            return [float(value) for value in values[:-1]], gripper_open_0to1

    def _read_corrected_angles(self) -> tuple[list[float], float]:
        raw_angles_deg, gripper_open_0to1 = self._read_raw_angles()
        corrected = [
            angle - offset
            for angle, offset in zip(raw_angles_deg, self._offset, strict=True)
        ]
        return corrected, gripper_open_0to1

    def _send_joint_targets(
        self,
        target_angles_deg: Sequence[float],
        *,
        gripper_open_0to1: float | None,
        interpolate: bool,
    ) -> None:
        current_angles_deg, current_gripper_0to1 = self._read_corrected_angles()
        desired_angles_deg = [float(angle) for angle in target_angles_deg]
        desired_gripper = (
            current_gripper_0to1 if gripper_open_0to1 is None else float(gripper_open_0to1)
        )
        if not 0.0 <= desired_gripper <= 1.0:
            raise ValueError("gripper_open_0to1 must be in [0, 1]")

        steps = int(self._cfg.get("motion_steps", 20)) if interpolate else 1
        alphas = np.linspace(0.0, 1.0, steps + 1)[1:] if interpolate else [1.0]

        for alpha in alphas:
            interp_angles = [
                current + (desired - current) * float(alpha)
                for current, desired in zip(current_angles_deg, desired_angles_deg, strict=True)
            ]
            interp_gripper = current_gripper_0to1 + (desired_gripper - current_gripper_0to1) * float(
                alpha
            )
            self._write_hardware(interp_angles, interp_gripper)
            if interpolate and steps > 1:
                time.sleep(0.5 / steps)

    def _write_hardware(
        self,
        corrected_angles_deg: Sequence[float],
        gripper_open_0to1: float,
    ) -> None:
        with self._lock:
            if self._sim is not None:
                self._sim.send_joint_targets(
                    MOTOR_NAMES,
                    corrected_angles_deg,
                    gripper_open_0to1,
                )
                return
            if self._arm is None:
                raise RuntimeError("Lerobo arm is not connected")

            action = {
                f"{motor_name}.pos": float(angle_deg + self._offset[index])
                for index, (motor_name, angle_deg) in enumerate(
                    zip(MOTOR_NAMES[:-1], corrected_angles_deg, strict=True)
                )
            }
            action["gripper.pos"] = float(gripper_open_0to1 * MAX_GRIPPER_ANGLE_DEG)
            self._arm.send_action(action)


def _make_header(frame_id: str) -> Header:
    now = time.time()
    sec = int(now)
    nanosec = int((now - sec) * 1e9)
    return Header(stamp=Time(sec=sec, nanosec=nanosec), frame_id=frame_id)


def _gripper_width_m(gripper_open_0to1: float, open_width_m: float) -> float:
    return float(max(0.0, min(1.0, gripper_open_0to1)) * open_width_m)


def _build_joint_state(
    joint_names: Sequence[str],
    arm_positions_rad: Sequence[float],
    gripper_open_0to1: float,
    *,
    gripper_joint_name: str = "gripper",
    gripper_open_width_m: float = 0.080,
    frame_id: str = "base_link",
) -> JointState:
    msg = JointState()
    msg.header = _make_header(frame_id)
    msg.name = list(joint_names)
    positions: list[float] = [float(value) for value in arm_positions_rad]
    if gripper_joint_name in joint_names:
        positions.append(_gripper_width_m(gripper_open_0to1, gripper_open_width_m))
    msg.position = positions
    return msg


def _build_end_pose(
    position_m: Sequence[float],
    quat_xyzw: Sequence[float],
    *,
    frame_id: str = "base_link",
) -> PoseStamped:
    msg = PoseStamped()
    msg.header = _make_header(frame_id)
    msg.pose = Pose(
        position=Point(
            x=float(position_m[0]),
            y=float(position_m[1]),
            z=float(position_m[2]),
        ),
        orientation=Quaternion(
            x=float(quat_xyzw[0]),
            y=float(quat_xyzw[1]),
            z=float(quat_xyzw[2]),
            w=float(quat_xyzw[3]),
        ),
    )
    return msg


def _parse_joint_command(
    msg: JointState,
    arm_joint_names: Sequence[str],
    *,
    gripper_joint_name: str = "gripper",
    gripper_open_width_m: float = 0.080,
) -> tuple[list[float] | None, float | None]:
    if not msg.name or not msg.position or len(msg.name) != len(msg.position):
        return None, None

    name_to_position = {
        name: float(position)
        for name, position in zip(msg.name, msg.position, strict=True)
    }
    arm_positions: list[float] = []
    for joint_name in arm_joint_names:
        if joint_name not in name_to_position:
            return None, None
        arm_positions.append(name_to_position[joint_name])

    gripper_open_0to1: float | None = None
    if gripper_joint_name in name_to_position and gripper_open_width_m > 0.0:
        gripper_open_0to1 = name_to_position[gripper_joint_name] / gripper_open_width_m
        gripper_open_0to1 = float(max(0.0, min(1.0, gripper_open_0to1)))

    return arm_positions, gripper_open_0to1


def _build_driver_cfg(cfg: dict) -> dict[str, Any]:
    package_root = _package_root(cfg)
    calibration_dir = cfg.get("calibration_dir") or str(
        package_root / "src/assets/calibration"
    )
    urdf_path = cfg.get("urdf_path") or str(
        package_root / "src/assets/urdf/lerobo/low_cost_robot.urdf"
    )
    lerobot_src = cfg.get("lerobot_src") or str(
        package_root / "src/vendor/lerobot/src"
    )
    arm_offset = cfg.get("arm_offset")
    if arm_offset is None:
        raise ValueError("arm_offset is required for Lerobo arm (five joint offsets in degrees)")

    return {
        "package_root": str(package_root),
        "lerobot_src": lerobot_src,
        "urdf_path": urdf_path,
        "calibration_dir": calibration_dir,
        "serial_port": cfg.get("serial_port", cfg.get("arm_port", "COM3")),
        "arm_backend": cfg.get("arm_backend", "real"),
        "arm_sim_host": cfg.get("arm_sim_host", "127.0.0.1"),
        "arm_sim_port": int(cfg.get("arm_sim_port", 8770)),
        "arm_sim_timeout_s": float(cfg.get("arm_sim_timeout_s", 10.0)),
        "connect_timeout_s": float(cfg.get("connect_timeout_s", 30.0)),
        "calibration_dir": calibration_dir,
        "robot_id": cfg.get("robot_id", "koch_follower"),
        "arm_offset": [float(value) for value in arm_offset],
        "motion_steps": int(cfg.get("motion_steps", 20)),
        "joint_names": list(cfg.get("joint_names", DEFAULT_JOINT_NAMES)),
        "publish_rate_hz": float(cfg.get("publish_rate_hz", 20)),
        "end_pose_frame": cfg.get("end_pose_frame", "base_link"),
        "joint_states_topic": cfg.get("joint_states_topic", "/roboarm/joint_states"),
        "end_pose_topic": cfg.get("end_pose_topic", "/roboarm/end_pose"),
        "joint_command_topic": cfg.get("joint_command_topic", "/roboarm/joint_command"),
        "gripper_open_width_m": float(cfg.get("gripper_open_width_m", 0.080)),
        "move_to_home_on_activate": bool(cfg.get("move_to_home_on_activate", True)),
        "allow_missing_hardware": bool(cfg.get("allow_missing_hardware", False)),
    }


provider = Primitive(id="roboarm_arm", namespace="robonix/primitive/arm")

_driver: LeroboDriver | None = None
_stop = threading.Event()
_pub_thread: threading.Thread | None = None
_cfg: dict = {}
_arm_joint_names: list[str] = []


@provider.on_init
def init(cfg: dict):
    global _driver, _cfg, _arm_joint_names
    try:
        _cfg = _build_driver_cfg(cfg)
    except ValueError as exc:
        return Err(str(exc))

    _arm_joint_names = [
        name for name in _cfg["joint_names"] if name != GRIPPER_JOINT_NAME
    ]
    if len(_arm_joint_names) != JOINT_COUNT:
        return Err(
            f"joint_names must contain exactly {JOINT_COUNT} arm joints plus one gripper entry"
        )

    provider.declare_ros2_topic(
        "robonix/primitive/arm/joint_states",
        _cfg["joint_states_topic"],
        qos="reliable",
    )
    provider.declare_ros2_topic(
        "robonix/primitive/arm/end_pose",
        _cfg["end_pose_topic"],
        qos="reliable",
    )
    provider.declare_ros2_topic(
        "robonix/primitive/arm/joint_command",
        _cfg["joint_command_topic"],
        qos="reliable",
    )

    provider.create_publisher(
        "robonix/primitive/arm/joint_states",
        topic=_cfg["joint_states_topic"],
        msg_type=JointState,
        qos="reliable",
        declare=False,
    )
    provider.create_publisher(
        "robonix/primitive/arm/end_pose",
        topic=_cfg["end_pose_topic"],
        msg_type=PoseStamped,
        qos="reliable",
        declare=False,
    )

    provider.create_subscription(
        "robonix/primitive/arm/joint_command",
        topic=_cfg["joint_command_topic"],
        msg_type=JointState,
        callback=_on_joint_command,
        qos="reliable",
        declare=False,
    )

    log.info("roboarm_arm initialized (hardware connect deferred to activate)")
    return Ok()


@provider.on_activate
def activate():
    global _pub_thread, _driver

    if _driver is None:
        try:
            _driver = LeroboDriver(_cfg)
            _driver.connect()
        except Exception as exc:
            _driver = None
            backend = _cfg.get("arm_backend", "real")
            target = _cfg.get("serial_port") if backend == "real" else (
                f"{_cfg.get('arm_sim_host')}:{_cfg.get('arm_sim_port')}"
            )
            if _cfg.get("allow_missing_hardware", False):
                log.warning(
                    "Lerobo arm unavailable (%s) at %s: %s — running in degraded mode",
                    backend,
                    target,
                    exc,
                )
                log.info("roboarm_arm activated (degraded, no joint control)")
                return Ok()
            return Err(f"Failed to connect Lerobo arm ({backend}) at {target}: {exc}")

    if _cfg.get("move_to_home_on_activate", True):
        try:
            _driver.move_to_home(gripper_open_0to1=1.0)
        except Exception as exc:
            log.warning("move_to_home on activate failed: %s", exc)

    _stop.clear()
    _pub_thread = threading.Thread(
        target=_publish_loop, name="roboarm_arm_pub", daemon=True
    )
    _pub_thread.start()
    log.info("roboarm_arm activated")
    return Ok()


@provider.on_deactivate
def deactivate():
    _stop.set()
    if _pub_thread is not None and _pub_thread.is_alive():
        _pub_thread.join(timeout=2.0)
    log.info("roboarm_arm deactivated")
    return Ok()


@provider.on_shutdown
def shutdown():
    global _driver

    deactivate()
    if _driver is not None:
        _driver.disconnect()
        _driver = None
    log.info("roboarm_arm shutdown complete")
    return Ok()


def _publish_loop() -> None:
    rate_hz = max(1.0, float(_cfg.get("publish_rate_hz", 20)))
    period_s = 1.0 / rate_hz
    frame_id = str(_cfg.get("end_pose_frame", "base_link"))
    gripper_width = float(_cfg.get("gripper_open_width_m", 0.080))

    while not _stop.is_set():
        loop_start = time.monotonic()
        if _driver is None:
            break
        try:
            arm_positions_rad, gripper_open_0to1, position_m, quat_xyzw = (
                _driver.read_joint_state()
            )
            joint_msg = _build_joint_state(
                _cfg["joint_names"],
                arm_positions_rad,
                gripper_open_0to1,
                gripper_joint_name=GRIPPER_JOINT_NAME,
                gripper_open_width_m=gripper_width,
                frame_id=frame_id,
            )
            pose_msg = _build_end_pose(
                position_m,
                quat_xyzw,
                frame_id=frame_id,
            )
            provider.emit("robonix/primitive/arm/joint_states", joint_msg)
            provider.emit("robonix/primitive/arm/end_pose", pose_msg)
        except Exception as exc:
            log.warning("Failed to publish arm state: %s", exc)

        elapsed = time.monotonic() - loop_start
        sleep_s = period_s - elapsed
        if sleep_s > 0:
            time.sleep(sleep_s)


def _on_joint_command(msg: JointState) -> None:
    if _driver is None:
        return
    arm_positions_rad, gripper_open_0to1 = _parse_joint_command(
        msg,
        _arm_joint_names,
        gripper_joint_name=GRIPPER_JOINT_NAME,
        gripper_open_width_m=float(_cfg.get("gripper_open_width_m", 0.080)),
    )
    if arm_positions_rad is None:
        log.warning("Ignoring joint_command with incomplete joint names")
        return
    try:
        _driver.write_joint_command(
            arm_positions_rad,
            gripper_open_0to1=gripper_open_0to1,
        )
    except Exception as exc:
        log.warning("Failed to apply joint_command: %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    provider.run()