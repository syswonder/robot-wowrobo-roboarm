from __future__ import annotations

import copy
import time
from queue import Queue
from typing import Any, Callable, Optional

import cv2
import numpy as np
from pydantic import TypeAdapter

from roboarm_core.config import get_config_value, resolve_asset
from roboarm_core.llm.dataclass import DetectedBox, DetectedFromLLM
from roboarm_core.llm.llm_detect import LLMDetect, json2box
from roboarm_core.vision.detect_viz import show_llm_detection, show_yolo_detection
from roboarm_core.vision.yolo_detect import detect_objects_in_frame, load_model

_YOLO_MODELS: dict[str, Any] = {}

_ALL_KEYWORDS = ("所有", "全部", "all", "every", "它们", "它们拿起", "都拿起", "都抓")


def _get_instruction_detect_backend() -> str:
    backend = str(
        get_config_value("instruction_detect_backend", "llm", raise_if_missing=False)
        or "llm"
    ).lower()
    if backend not in {"llm", "yolo"}:
        raise ValueError(
            f"instruction_detect_backend must be 'llm' or 'yolo', got: {backend}"
        )
    return backend


def _load_yolo_models() -> list[Any]:
    model_paths = [
        resolve_asset(path)
        for path in get_config_value("classification_YOLO_model_path", [])
    ]
    models: list[Any] = []
    for model_path in model_paths:
        key = str(model_path)
        if key not in _YOLO_MODELS:
            _YOLO_MODELS[key] = load_model(key)
        models.append(_YOLO_MODELS[key])
    return models


def _run_yolo_detections(frame: cv2.typing.MatLike) -> list[tuple]:
    conf_thres = get_config_value("default_conf_thres")
    detections: list[tuple] = []
    for model in _load_yolo_models():
        detections.extend(
            detect_objects_in_frame(model, frame, conf_thres=conf_thres)
        )
    return detections


def _yolo_tuple_to_box(detection: tuple) -> DetectedBox:
    (u, v, w, h, r), score, _class_id, class_name = detection
    return DetectedBox(
        class_name=class_name,
        box_center_x=float(u),
        box_center_y=float(v),
        box_width=float(w),
        box_height=float(h),
        box_rotation_deg=float(np.rad2deg(r)),
        confidence=float(score),
    )


def _instruction_implies_all(instruction: str) -> bool:
    lower = instruction.lower()
    return any(keyword in lower for keyword in _ALL_KEYWORDS)


def _score_detection_for_instruction(
    class_name: str,
    instruction: str,
    place_pos: dict,
) -> float:
    lower_inst = instruction.lower()
    lower_cls = class_name.lower()
    score = 0.0
    if lower_cls in lower_inst:
        score += 2.0
    for token in lower_inst.replace("，", " ").replace(",", " ").split():
        if token and token in lower_cls:
            score += 1.5
    for block_name, pos_data in place_pos.items():
        if block_name.lower() in lower_cls:
            score += 1.0
        for keyword in pos_data.get("keywords", []):
            kw = str(keyword).lower()
            if kw in lower_inst and (kw in lower_cls or block_name.lower() in lower_cls):
                score += 3.0
    return score


def _select_yolo_boxes(
    detections: list[tuple],
    instruction: str,
) -> list[DetectedBox]:
    if not detections:
        return []
    boxes = [_yolo_tuple_to_box(det) for det in detections]
    if _instruction_implies_all(instruction):
        return boxes
    place_pos = get_config_value("place_pos", default={}, raise_if_missing=False)
    scored = [
        (_score_detection_for_instruction(box.class_name, instruction, place_pos), box)
        for box in boxes
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored[0][0] <= 0:
        best = max(boxes, key=lambda box: box.confidence or 0.0)
        return [best]
    best_score = scored[0][0]
    return [box for score, box in scored if score >= best_score]


def _resolve_class_place_pos(
    box: DetectedBox,
    place_pos: dict,
    *,
    target_x: float,
    target_y: float,
) -> list[float | int]:
    class_place_pos_data = place_pos.get(box.class_name)
    if class_place_pos_data is None or "pos" not in class_place_pos_data:
        for pos_data in place_pos.values():
            for keyword in pos_data.get("keywords", []):
                if keyword.lower() in box.class_name.lower():
                    class_place_pos_data = pos_data
                    break
            if class_place_pos_data is not None:
                break
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
    return class_place_pos


def _grasp_box(
    arm: Any,
    box: DetectedBox,
    *,
    offset: float,
    place_pos: dict,
    queue_output: Queue,
) -> bool:
    queue_output.put(box)
    target_x, target_y = arm.pixel2pos(box.box_center_x, box.box_center_y)
    gripper_angle_rad = arm.gripper_angle_by_longer(
        box.box_center_x,
        box.box_center_y,
        box.box_width,
        box.box_height,
        box.box_rotation_deg,
    )
    class_place_pos = _resolve_class_place_pos(
        box,
        place_pos,
        target_x=target_x,
        target_y=target_y,
    )
    return arm.catch_and_place(
        target_x + offset * np.cos(gripper_angle_rad),
        target_y + offset * np.sin(-gripper_angle_rad),
        gripper_angle_rad,
        class_place_pos,
    )


def _prepare_arm_for_grasp(arm: Any) -> None:
    default_gripper_aside_pos = get_config_value(
        "default_gripper_aside_pos", raise_if_missing=False
    )
    if default_gripper_aside_pos is not None:
        arm.move_to(default_gripper_aside_pos)
    time.sleep(0.5)


def _catch_by_instruction_llm(
    frame: cv2.typing.MatLike,
    instruction: str,
    queue_output: Queue,
    arm: Any,
) -> tuple[list[DetectedBox], list[bool]]:
    llm_detect = LLMDetect()
    response_task = llm_detect.detect_frame(
        frame,
        prompt_key="user_instruction_prompt",
        replace_map={"{user_instruction}": instruction},
        schema=TypeAdapter(DetectedFromLLM).json_schema(),
    )
    boxes: list[DetectedBox] = []
    results: list[bool] = []
    if not response_task:
        return boxes, results

    place_pos = get_config_value("place_pos", default={}, raise_if_missing=False)
    offset = get_config_value("catch_offset")

    while True:
        response, done = llm_detect.llm_api.await_task(response_task, blocking=False)
        if response:
            box = json2box(response, img_w=frame.shape[1], img_h=frame.shape[0])
            if box:
                boxes.append(box)
                results.append(
                    _grasp_box(
                        arm,
                        box,
                        offset=offset,
                        place_pos=place_pos,
                        queue_output=queue_output,
                    )
                )
        status = ["instruction: " + instruction]
        if boxes:
            status.append(f"detected: {len(boxes)}")
        elif not done:
            status.append("waiting for LLM...")
        show_llm_detection(frame, boxes, status_lines=status)
        if done:
            break
    return boxes, results


def _catch_by_instruction_yolo(
    frame: cv2.typing.MatLike,
    instruction: str,
    queue_output: Queue,
    arm: Any,
) -> tuple[list[DetectedBox], list[bool]]:
    detections = _run_yolo_detections(frame)
    boxes = _select_yolo_boxes(detections, instruction)
    show_yolo_detection(
        frame,
        detections,
        status_lines=[
            f"instruction: {instruction}",
            f"selected: {len(boxes)} / {len(detections)}",
        ],
    )
    place_pos = get_config_value("place_pos", default={}, raise_if_missing=False)
    offset = get_config_value("catch_offset")
    results: list[bool] = []
    for box in boxes:
        results.append(
            _grasp_box(
                arm,
                box,
                offset=offset,
                place_pos=place_pos,
                queue_output=queue_output,
            )
        )
    return boxes, results


def catch_by_instruction(
    frame: cv2.typing.MatLike,
    instruction: str,
    queue_output: Queue,
    success_callback: Optional[Callable[[], None]] = None,
    arm: Any = None,
) -> dict[str, Any]:
    if arm is None:
        raise RuntimeError("arm instance is required for Robonix grasp")

    backend = _get_instruction_detect_backend()
    boxes: list[DetectedBox] = []
    grasp_results: list[bool] = []

    try:
        _prepare_arm_for_grasp(arm)
        if backend == "yolo":
            boxes, grasp_results = _catch_by_instruction_yolo(
                frame, instruction, queue_output, arm
            )
        else:
            boxes, grasp_results = _catch_by_instruction_llm(
                frame, instruction, queue_output, arm
            )
        if any(grasp_results) and success_callback:
            success_callback()
    except Exception as exc:
        print("Exception:", exc)
        return {
            "status": "failed",
            "reason": str(exc),
            "instruction": instruction,
            "method": backend,
            "grasp_success": False,
        }

    if not boxes:
        return {
            "status": "failed",
            "reason": "未检测到目标物体",
            "instruction": instruction,
            "method": backend,
            "grasp_success": False,
        }

    success_count = sum(1 for ok in grasp_results if ok)
    return {
        "status": "success" if success_count > 0 else "failed",
        "target": ", ".join(box.class_name for box in boxes),
        "instruction": instruction,
        "method": backend,
        "grasp_success": success_count > 0,
        "grasped_count": success_count,
        "detected_count": len(boxes),
    }
