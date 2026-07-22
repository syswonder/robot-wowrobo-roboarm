"""YOLO 检测 + 分类抓取逻辑（与 tools/lib/grasp_pipeline 对齐）。"""

from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np
from roboarm_core.arm.arm_base import Arm
from roboarm_core.config import get_config_value


def _resolve_place_pos(
    class_name: str,
    place_pos: dict[str, Any],
    target_x: float,
    target_y: float,
) -> list[float]:
    class_place_pos_data = place_pos.get(class_name)
    if class_place_pos_data is None or "pos" not in class_place_pos_data:
        return [target_x, target_y]
    class_place_pos = copy.deepcopy(class_place_pos_data["pos"])
    for index, ref in enumerate(class_place_pos):
        if ref == "x":
            class_place_pos[index] = target_x
        elif ref == "-x":
            class_place_pos[index] = -target_x
        elif ref == "y":
            class_place_pos[index] = target_y
        elif ref == "-y":
            class_place_pos[index] = -target_y
    return [float(v) for v in class_place_pos]


def grasp_detections(
    arm: Arm,
    detections: list,
    *,
    on_progress: Callable[[str], None] | None = None,
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
        class_place_pos = _resolve_place_pos(
            class_name, place_pos, target_x, target_y
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
