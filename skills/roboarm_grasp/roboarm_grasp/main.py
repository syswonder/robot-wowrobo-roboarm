#!/usr/bin/env python3
"""roboarm_grasp — 通过 Atlas 调用机械臂/相机原语，封装积木分类与 LLM 抓取。"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from robonix_api import ATLAS, Err, Ok, Skill
from roboarm_core.arm.robonix_arm import RobonixArm
from roboarm_core.config import init_config
from roboarm_core.grasp_pipeline import grasp_detections, move_gripper_aside, run_classify_yolo_detection
from roboarm_core.llm.catch_by_llm import catch_by_instruction
from roboarm_core.cv2_display import destroy_all_windows
from roboarm_core.vision.detect_viz import is_dev_mode
from sensor_msgs.msg import Image

log = logging.getLogger("roboarm_grasp")

skill = Skill(id="roboarm_grasp", namespace="robonix/skill/roboarm_grasp")

_cfg: dict[str, Any] = {}
_pkg_root: Path | None = None
_arm: RobonixArm | None = None
_latest_bgr: np.ndarray | None = None
_frame_lock = threading.Lock()
_rgb_sub = None


def _resolve_pkg_root(cfg: dict) -> Path:
    raw = cfg.get("package_root") or os.environ.get("RBNX_PACKAGE_ROOT")
    if raw:
        path = Path(str(raw)).resolve()
        if not path.is_dir():
            raise ValueError(f"package_root does not exist: {path}")
        return path
    return Path(__file__).resolve().parent.parent


def _resolve_topic(contract_id: str, *, timeout_s: float = 30.0) -> str:
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
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
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


def _setup_arm() -> None:
    global _arm
    assert _pkg_root is not None
    assets_root = Path(_cfg.get("assets_dir") or _pkg_root / "assets")
    joint_command_topic = _resolve_topic("robonix/primitive/arm/joint_command")
    joint_states_topic = _resolve_topic("robonix/primitive/arm/joint_states")
    _arm = RobonixArm(
        joint_names=_cfg.get(
            "joint_names",
            ["joint1", "joint2", "joint3", "joint4", "joint5", "gripper"],
        ),
        arm_offset=_cfg["arm_offset"],
        assets_root=assets_root,
        joint_command_topic=joint_command_topic,
        joint_states_topic=joint_states_topic,
        gripper_open_width_m=float(_cfg.get("gripper_open_width_m", 0.080)),
        motion_steps=int(_cfg.get("motion_steps", 20)),
    )
    _arm.connect()
    _arm.move_to_home(gripper_open_0to1=1, block_until_reach=True)


def _setup_camera() -> None:
    global _rgb_sub
    from robonix_api.ros import RosBackend

    rgb_topic = _resolve_topic("robonix/primitive/camera/rgb")
    _rgb_sub = RosBackend.get().create_subscription(
        Image, rgb_topic, _on_rgb, "reliable"
    )
    if not RosBackend.get().wait_for_topic(rgb_topic, Image, timeout_s=30.0):
        raise TimeoutError(f"Timed out waiting for camera rgb on {rgb_topic}")


@skill.on_init
def init(cfg: dict):
    global _cfg, _pkg_root
    _cfg = dict(cfg)

    try:
        _pkg_root = _resolve_pkg_root(_cfg)
    except ValueError as exc:
        return Err(str(exc))

    if _cfg.get("arm_offset") is None:
        return Err("arm_offset is required in skill config")

    config_yaml = _cfg.get("config_yaml") or str(_pkg_root / "config" / "config.yaml")
    assets_dir = _cfg.get("assets_dir") or str(_pkg_root / "assets")
    try:
        init_config(config_yaml, assets_root=assets_dir)
    except (FileNotFoundError, OSError) as exc:
        return Err(f"Failed to load config: {exc}")

    arm_id = str(_cfg.get("arm_provider_id", "roboarm_arm"))
    camera_id = str(_cfg.get("camera_provider_id", "orbbec_camera_roboarm"))
    if not ATLAS.find_capability(
        contract_id="robonix/primitive/arm/joint_command", provider_id=arm_id
    ):
        log.warning("roboarm_arm joint_command not found yet; will retry on activate")
    if not ATLAS.find_capability(
        contract_id="robonix/primitive/camera/rgb", provider_id=camera_id
    ):
        log.warning("orbbec camera rgb not found yet; will retry on activate")

    log.info("roboarm_grasp initialized (package_root=%s)", _pkg_root)
    return Ok()


@skill.on_activate
def activate():
    try:
        _setup_arm()
        _setup_camera()
    except Exception as exc:
        return Err(f"Failed to connect primitives: {exc}")
    log.info("roboarm_grasp activated")
    return Ok()


@skill.on_deactivate
def deactivate():
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
    if is_dev_mode():
        destroy_all_windows()
    log.info("roboarm_grasp deactivated")
    return Ok()


try:
    from roboarm_grasp_mcp import (  # type: ignore
        ClassifyAndGrasp_Request,
        ClassifyAndGrasp_Response,
        GraspByInstruction_Request,
        GraspByInstruction_Response,
    )
    from std_msgs_mcp import String
except ImportError:  # pragma: no cover
    ClassifyAndGrasp_Request = ClassifyAndGrasp_Response = None  # type: ignore
    GraspByInstruction_Request = GraspByInstruction_Response = None  # type: ignore
    String = None  # type: ignore


if ClassifyAndGrasp_Request is not None:

    @skill.mcp("robonix/skill/roboarm_grasp/classify_and_grasp")
    def classify_and_grasp(_req: ClassifyAndGrasp_Request) -> ClassifyAndGrasp_Response:
        if _arm is None:
            raise RuntimeError("skill is not activated")

        frame = _get_bgr_frame()
        if frame is None:
            return ClassifyAndGrasp_Response(
                success=False,
                message=String(data="无法获取相机画面"),
            )

        move_gripper_aside(_arm)
        detections = run_classify_yolo_detection(frame)

        if not detections:
            return ClassifyAndGrasp_Response(
                success=True,
                message=String(data="未检测到目标"),
            )

        ok_n, fail_n, details = grasp_detections(_arm, detections)
        summary = f"检测 {len(detections)} 个，成功 {ok_n}，失败 {fail_n}； " + "; ".join(
            details
        )
        return ClassifyAndGrasp_Response(
            success=fail_n == 0 and ok_n > 0,
            message=String(data=summary),
        )


if GraspByInstruction_Request is not None:

    @skill.mcp("robonix/skill/roboarm_grasp/grasp_by_instruction")
    def grasp_by_instruction_tool(
        req: GraspByInstruction_Request,
    ) -> GraspByInstruction_Response:
        from queue import Queue

        if _arm is None:
            raise RuntimeError("skill is not activated")

        instruction = (req.instruction.data or "").strip()
        if not instruction:
            return GraspByInstruction_Response(
                success=False,
                message=String(data="instruction 不能为空"),
            )

        frame = _get_bgr_frame()
        if frame is None:
            return GraspByInstruction_Response(
                success=False,
                message=String(data="无法获取相机画面"),
            )

        result = catch_by_instruction(frame, instruction, Queue(), arm=_arm)
        success = result.get("status") == "success" and result.get("grasp_success")
        target = result.get("target", "")
        method = result.get("method", "llm")
        msg = (
            f"检测: {method}; 指令: {instruction}; 目标: {target or '无'}; "
            f"结果: {result.get('status')}"
        )
        if result.get("detected_count") is not None:
            msg += (
                f"; 检测 {result['detected_count']} 个，"
                f"成功抓取 {result.get('grasped_count', 0)} 个"
            )
        if result.get("reason"):
            msg += f"; {result['reason']}"
        return GraspByInstruction_Response(
            success=bool(success),
            message=String(data=msg),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    skill.run()
