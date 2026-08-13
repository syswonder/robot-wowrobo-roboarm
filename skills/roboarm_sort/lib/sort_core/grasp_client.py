"""直接调用 roboarm_core 的 grasp/place 函数（不经 MCP）。"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Any

import cv2
import numpy as np
from robonix_api import ATLAS
from roboarm_core.arm.robonix_arm import RobonixArm
from roboarm_core.config import init_config as init_grasp_config
from roboarm_core.llm.catch_by_llm import (
    format_grasp_message,
    format_place_message,
    grasp_by_instruction as _grasp_by_instruction,
)
from roboarm_core.llm.catch_by_llm import (
    place_by_instruction as _place_by_instruction,
)
from roboarm_core.llm.catch_by_llm import _place_at
from sensor_msgs.msg import Image

log = logging.getLogger("roboarm_sort.grasp_client")

_arm: RobonixArm | None = None
_latest_bgr: np.ndarray | None = None
_frame_lock = threading.Lock()
_rgb_sub = None
_grasp_config_loaded = False


def _resolve_config_path(raw: str, pkg_root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = (
        pkg_root / path,
        pkg_root.parent / path,
        pkg_root.parent.parent / path,
        Path.cwd() / path,
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    return (pkg_root / path).resolve()


def _resolve_grasp_paths(cfg: dict, pkg_root: Path) -> tuple[Path, Path]:
    raw_yaml = cfg.get("grasp_config_yaml")
    if raw_yaml:
        config_yaml = _resolve_config_path(str(raw_yaml), pkg_root)
    else:
        config_yaml = (pkg_root.parent / "roboarm_grasp" / "config" / "config.yaml").resolve()
    if not config_yaml.is_file():
        raise FileNotFoundError(f"grasp config not found: {config_yaml}")

    raw_assets = cfg.get("grasp_assets_dir")
    if raw_assets:
        assets_path = Path(str(raw_assets))
        if assets_path.is_absolute():
            assets_root = assets_path
        else:
            assets_root = None
            for base in (pkg_root, pkg_root.parent, pkg_root.parent.parent, Path.cwd()):
                candidate = (base / assets_path).resolve()
                if candidate.is_dir():
                    assets_root = candidate
                    break
            if assets_root is None:
                assets_root = (pkg_root / assets_path).resolve()
    else:
        assets_root = config_yaml.parent.parent / "assets"
    return config_yaml, assets_root


def load_grasp_config(cfg: dict, pkg_root: Path) -> None:
    global _grasp_config_loaded
    config_yaml, assets_root = _resolve_grasp_paths(cfg, pkg_root)
    init_grasp_config(str(config_yaml), assets_root=str(assets_root))
    _grasp_config_loaded = True
    log.info("roboarm_grasp config loaded: %s", config_yaml)


def _resolve_topic(skill: Any, contract_id: str, *, timeout_s: float = 30.0) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        caps = ATLAS.find_capability(contract_id=contract_id, transport="ros2")
        if caps:
            with skill.connect_capability(caps[0], contract_id, "ros2") as ch:
                if ch.endpoint:
                    return ch.endpoint
        time.sleep(1.0)
    raise TimeoutError(f"Atlas 未找到 ROS2 能力: {contract_id}")


def _ros_rgb_to_bgr(msg: Image) -> np.ndarray:
    if msg.encoding == "rgb8":
        rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if msg.encoding == "bgr8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        ).copy()
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def _on_rgb(msg: Image) -> None:
    global _latest_bgr
    try:
        bgr = _ros_rgb_to_bgr(msg)
    except Exception as exc:
        log.warning("Failed to decode camera frame: %s", exc)
        return
    with _frame_lock:
        _latest_bgr = bgr


def _get_bgr_frame(timeout_s: float = 5.0) -> np.ndarray | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with _frame_lock:
            if _latest_bgr is not None:
                return _latest_bgr.copy()
        time.sleep(0.05)
    return None


def _setup_arm(skill: Any, cfg: dict, assets_root: Path) -> None:
    global _arm
    if cfg.get("arm_offset") is None:
        raise ValueError("arm_offset is required in roboarm_sort skill config")

    joint_command_topic = _resolve_topic(skill, "robonix/primitive/arm/joint_command")
    joint_states_topic = _resolve_topic(skill, "robonix/primitive/arm/joint_states")
    _arm = RobonixArm(
        joint_names=cfg.get(
            "joint_names",
            ["joint1", "joint2", "joint3", "joint4", "joint5", "gripper"],
        ),
        arm_offset=cfg["arm_offset"],
        assets_root=assets_root,
        joint_command_topic=joint_command_topic,
        joint_states_topic=joint_states_topic,
        gripper_open_width_m=float(cfg.get("gripper_open_width_m", 0.080)),
        motion_steps=int(cfg.get("motion_steps", 20)),
    )
    _arm.connect()
    _arm.move_to_home(gripper_open_0to1=1, block_until_reach=True)


def _setup_camera(skill: Any) -> None:
    global _rgb_sub
    from robonix_api.ros import RosBackend

    rgb_topic = _resolve_topic(skill, "robonix/primitive/camera/rgb")
    _rgb_sub = RosBackend.get().create_subscription(
        Image, rgb_topic, _on_rgb, "reliable"
    )
    if not RosBackend.get().wait_for_topic(rgb_topic, Image, timeout_s=30.0):
        raise TimeoutError(f"Timed out waiting for camera rgb on {rgb_topic}")


def connect_grasp_runtime(skill: Any, cfg: dict, pkg_root: Path) -> None:
    if not _grasp_config_loaded:
        load_grasp_config(cfg, pkg_root)
    config_yaml, assets_root = _resolve_grasp_paths(cfg, pkg_root)
    _setup_arm(skill, cfg, assets_root)
    _setup_camera(skill)
    log.info("grasp runtime connected (config=%s)", config_yaml)


def close_grasp_client() -> None:
    global _arm, _rgb_sub, _latest_bgr
    if _arm is not None:
        try:
            _arm.move_to_home(gripper_open_0to1=1, block_until_reach=True)
            _arm.disconnect_arm()
        except Exception as exc:
            log.warning("Error while disconnecting arm: %s", exc)
        _arm = None
    _rgb_sub = None
    with _frame_lock:
        _latest_bgr = None


def _format_grasp_result(
    instruction: str, result: dict[str, Any]
) -> tuple[bool, str, str]:
    success = result.get("status") == "success" and result.get("grasp_success")
    target = str(result.get("target", "") or "")
    return bool(success), format_grasp_message(instruction, result), target


def _format_place_result(instruction: str, result: dict[str, Any]) -> tuple[bool, str]:
    success = result.get("status") == "success" and result.get("place_success")
    return bool(success), format_place_message(instruction, result)


def _mcp_style_response(
    *, success: bool, message: str, class_name: str = ""
) -> dict[str, Any]:
    resp: dict[str, Any] = {
        "success": success,
        "message": {"data": message},
    }
    if class_name:
        resp["class_name"] = {"data": class_name}
    return resp


def grasp_by_instruction(instruction: str) -> dict[str, Any]:
    if _arm is None:
        raise RuntimeError("grasp runtime 未就绪，请先 activate roboarm_sort")
    result = _grasp_by_instruction(
        None, instruction, Queue(), arm=_arm, get_frame=_get_bgr_frame
    )
    ok, msg, class_name = _format_grasp_result(instruction, result)
    log.info(
        "grasp_by_instruction(%r) -> success=%s msg=%r",
        instruction,
        ok,
        msg[:120],
    )
    return _mcp_style_response(success=ok, message=msg, class_name=class_name)


def place_by_instruction(
    instruction: str,
    *,
    class_name: str = "",
    pos: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    if _arm is None:
        raise RuntimeError("grasp runtime 未就绪，请先 activate roboarm_sort")

    if pos is not None:
        place_key = (instruction or class_name or "sort_slot").strip()
        place_pos = {place_key: {"pos": list(pos)}}
        try:
            ok = _place_at(
                _arm,
                instruction=place_key,
                class_name=class_name,
                place_pos=place_pos,
            )
        except Exception as exc:
            log.error("place_by_instruction 异常: %s", exc, exc_info=True)
            result = {
                "status": "failed",
                "failure_stage": "exception",
                "reason": str(exc),
                "instruction": instruction,
                "target": class_name,
                "place_success": False,
            }
        else:
            result = {
                "status": "success" if ok else "failed",
                "failure_stage": None if ok else "place",
                "target": class_name,
                "instruction": instruction,
                "place_success": ok,
                "reason": None if ok else "机械臂未能成功放置目标",
            }
    else:
        result = _place_by_instruction(instruction, _arm, class_name=class_name)

    ok, msg = _format_place_result(instruction, result)
    if pos is not None:
        msg += f"; pos={list(pos)}"
    log.info(
        "place_by_instruction(%r, class_name=%r, pos=%r) -> success=%s",
        instruction,
        class_name,
        pos,
        ok,
    )
    return _mcp_style_response(success=ok, message=msg)


def _nested_str(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("data", "") or "")
    return str(value or "")


def grasp_success(resp: dict[str, Any]) -> bool:
    return bool(resp.get("success"))


def place_success(resp: dict[str, Any]) -> bool:
    return bool(resp.get("success"))


def grasp_class_name(resp: dict[str, Any]) -> str:
    return _nested_str(resp.get("class_name"))


def response_message(resp: dict[str, Any]) -> str:
    return _nested_str(resp.get("message"))
