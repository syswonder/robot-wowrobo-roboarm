"""YOLO 检测 + 分类抓取逻辑（roboarm_grasp skill）。"""
#暂时弃用
from __future__ import annotations

from typing import Any

import numpy as np
from roboarm_core.arm.arm_base import Arm
from roboarm_core.config import get_config_value, resolve_asset
from roboarm_core.place_pos import resolve_place_pos
from roboarm_core.vision.detect_viz import show_yolo_detection
from roboarm_core.vision.yolo_detect import detect_objects_in_frame, load_model

Detection = tuple[tuple[float, float, float, float, float], float, int, str]


def load_yolo_models() -> list[Any]:
    model_paths = [
        resolve_asset(path)
        for path in get_config_value("classification_YOLO_model_path", [])
    ]
    if not model_paths:
        raise RuntimeError("classification_YOLO_model_path 未配置")
    return [load_model(str(path)) for path in model_paths]


def detect_all(models: list[Any], frame: np.ndarray) -> list[Detection]:
    conf_thres = get_config_value("default_conf_thres")
    detections: list[Detection] = []
    for model in models:
        detections.extend(
            detect_objects_in_frame(model, frame, conf_thres=conf_thres)
        )
    return detections


def grasp_detections(
    arm: Arm,
    detections: list[Detection],
    *,
    on_progress: Any | None = None,
) -> tuple[int, int, list[str]]:
    place_pos = get_config_value("place_pos", default={}, raise_if_missing=False)
    place_distance_threshold = float(
        get_config_value("place_distance_threshold", default=0, raise_if_missing=False)
    )
    offset = float(get_config_value("catch_offset"))
    success_count = 0
    fail_count = 0
    details: list[str] = []

    for (u, v, w, h, r), score, _class_id, class_name in detections:
        angle_deg = float(np.rad2deg(r))
        target_x, target_y = arm.pixel2pos(u, v)
        gripper_angle_rad = arm.gripper_angle_by_longer(u, v, w, h, angle_deg)
        class_place_pos = resolve_place_pos(
            class_name,
            place_pos=place_pos,
            target_x=target_x,
            target_y=target_y,
        )

        if (
            place_distance_threshold > 0
            and np.linalg.norm(
                np.array(class_place_pos) - np.array([target_x, target_y])
            )
            < place_distance_threshold
        ):
            detail = f"{class_name}: skipped (too close to place pos)"
            details.append(detail)
            if on_progress is not None:
                on_progress(detail)
            continue

        if on_progress is not None:
            on_progress(f"{class_name}: grasping...")

        ok = arm.catch_and_place(
            target_x + offset * np.cos(gripper_angle_rad),
            target_y + offset * np.sin(-gripper_angle_rad),
            gripper_angle_rad,
            class_place_pos,
        )
        if ok:
            success_count += 1
            detail = f"{class_name}: OK"
        else:
            fail_count += 1
            detail = f"{class_name}: FAIL"
        details.append(detail)
        if on_progress is not None:
            on_progress(detail)

    return success_count, fail_count, details


def move_gripper_aside(arm: Arm) -> None:
    aside = get_config_value("default_gripper_aside_pos", raise_if_missing=False)
    if aside:
        arm.move_to(aside, 1, block_until_reach=True)

def _get_save_path() -> str:
    save_flag = get_config_value("save_img", raise_if_missing=False)
    if save_flag:
        return "../../grasp_debug.jpg"
    return None

def run_classify_yolo_detection(frame: np.ndarray) -> list[Detection]:
    detections = detect_all(load_yolo_models(), frame)
    save_path = _get_save_path()
    show_yolo_detection(
        frame,
        detections,
        status_lines=[f"detected: {len(detections)}"],
        save_path=save_path,
    )
    return detections
