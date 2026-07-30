from __future__ import annotations

import logging
import time
from queue import Queue
from typing import Any, Callable, Optional

import cv2
import numpy as np
from pydantic import TypeAdapter

from roboarm_core.config import get_config_value, resolve_asset
from roboarm_core.llm.dataclass import DetectedBox, DetectedFromLLM, InstructionDetectResponse
from roboarm_core.llm.llm_detect import LLMDetect
from roboarm_core.place_pos import resolve_place_pos
from roboarm_core.vision.detect_viz import show_llm_detection, show_yolo_detection
from roboarm_core.vision.mobile_sam_refine import (
    RefineResult,
    SamRefineDebug,
    parse_llm_detections,
    refine_detection,
)
from roboarm_core.vision.yolo_detect import detect_objects_in_frame, load_model

log = logging.getLogger("roboarm_grasp.catch_by_llm")

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


def _yolo_tuple_to_detected_from_llm(detection: tuple, frame: cv2.typing.MatLike) -> DetectedFromLLM:
    (u, v, w, h, _r), _score, _class_id, class_name = detection
    img_h, img_w = frame.shape[:2]
    return DetectedFromLLM(
        id=0,
        class_name=class_name,
        box_center_x=float(u) / img_w,
        box_center_y=float(v) / img_h,
        box_width=float(w) / img_w,
        box_height=float(h) / img_h,
    )


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


def _select_yolo_detections(
    detections: list[tuple],
    instruction: str,
) -> list[tuple]:
    if not detections:
        return []
    if _instruction_implies_all(instruction):
        return detections
    place_pos = get_config_value("place_pos", default={}, raise_if_missing=False)
    scored = [
        (
            _score_detection_for_instruction(
                _yolo_tuple_to_box(det).class_name, instruction, place_pos
            ),
            det,
        )
        for det in detections
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored[0][0] <= 0:
        best_det = max(
            detections,
            key=lambda det: _yolo_tuple_to_box(det).confidence or 0.0,
        )
        return [best_det]
    best_score = scored[0][0]
    return [det for score, det in scored if score >= best_score]


def _catch_box(
    arm: Any,
    box: DetectedBox,
    *,
    offset: float,
    queue_output: Queue,
) -> tuple[bool, float, float, float]:
    queue_output.put(box)
    target_x, target_y = arm.pixel2pos(box.box_center_x, box.box_center_y)
    gripper_angle_rad = arm.gripper_angle_by_longer(
        box.box_center_x,
        box.box_center_y,
        box.box_width,
        box.box_height,
        box.box_rotation_deg,
    )
    ok = arm.catch(
        target_x + offset * np.cos(gripper_angle_rad),
        target_y + offset * np.sin(-gripper_angle_rad),
        gripper_angle_rad,
    )
    return ok, target_x, target_y, gripper_angle_rad


def _place_at(
    arm: Any,
    *,
    instruction: str,
    class_name: str,
    place_pos: dict,
) -> bool:
    class_place_pos = resolve_place_pos(
        instruction=instruction,
        class_name=class_name,
        place_pos=place_pos,
    )
    if len(class_place_pos) == 2:
        place_x, place_y = class_place_pos
        place_z = arm.desktop_height
    elif len(class_place_pos) == 3:
        place_x, place_y, place_z = class_place_pos
    else:
        return False
    if not arm.place(place_x, place_y, place_z):
        arm.move_to_home(gripper_open_0to1=1)
        return False
    arm.move_to_home(gripper_open_0to1=1)
    return True


def _prepare_arm_for_grasp(arm: Any) -> None:
    default_gripper_aside_pos = get_config_value(
        "default_gripper_aside_pos", raise_if_missing=False
    )
    if default_gripper_aside_pos is not None:
        arm.move_to(default_gripper_aside_pos)
    time.sleep(0.5)


def _refine_and_catch_first(
    frame: cv2.typing.MatLike,
    detections: list[DetectedFromLLM],
    *,
    arm: Any,
    offset: float,
    instruction: str,
    queue_output: Queue,
    save_path: str | None = None,
) -> tuple[DetectedBox | None, bool, list[SamRefineDebug]]:
    img_w, img_h = frame.shape[1], frame.shape[0]
    sam_debug_list: list[SamRefineDebug] = []
    refined_boxes: list[DetectedBox] = []

    for detection in detections:
        refine_result: RefineResult = refine_detection(
            frame,
            detection,
            img_w=img_w,
            img_h=img_h,
        )
        if refine_result.sam_debug is not None:
            sam_debug_list.append(refine_result.sam_debug)
        box = refine_result.box
        if not box:
            continue
        refined_boxes.append(box)
        ok, target_x, target_y, _gripper_angle_rad = _catch_box(
            arm, box, offset=offset, queue_output=queue_output
        )
        status = ["instruction: " + instruction, f"refined: {len(refined_boxes)}"]
        if ok:
            status.append(f"caught: {box.class_name}")
            show_llm_detection(
                frame,
                [box],
                sam_debug_list=sam_debug_list or None,
                status_lines=status,
                save_path=save_path,
            )
            return box, True, sam_debug_list
        status.append(f"catch failed: {box.class_name}")

    show_llm_detection(
        frame,
        refined_boxes,
        sam_debug_list=sam_debug_list or None,
        status_lines=[
            "instruction: " + instruction,
            "no successful catch after refine",
        ],
        save_path=save_path,
    )
    return None, False, sam_debug_list


def _grasp_by_instruction_llm(
    frame: cv2.typing.MatLike,
    instruction: str,
    queue_output: Queue,
    arm: Any,
    save_path: str | None = None,
) -> tuple[DetectedBox | None, bool]:
    llm_detect = LLMDetect()
    response_task = llm_detect.detect_frame(
        frame,
        prompt_key="user_instruction_prompt",
        replace_map={"{user_instruction}": instruction},
        schema=TypeAdapter(InstructionDetectResponse).json_schema(),
    )
    offset = get_config_value("catch_offset")
    processed = False
    box: DetectedBox | None = None
    caught = False
    sam_debug_list: list[SamRefineDebug] = []

    while True:
        response, done = llm_detect.llm_api.await_task(response_task, blocking=False)
        if response and not processed:
            detections = parse_llm_detections(response)
            box, caught, sam_debug_list = _refine_and_catch_first(
                frame,
                detections,
                arm=arm,
                offset=offset,
                instruction=instruction,
                queue_output=queue_output,
                save_path=save_path,
            )
            processed = True
        elif not processed:
            status = ["instruction: " + instruction, "waiting for LLM..."]
            show_llm_detection(
                frame,
                [],
                status_lines=status,
                save_path=save_path,
            )
        if done:
            break
    return box, caught


def _grasp_by_instruction_yolo(
    frame: cv2.typing.MatLike,
    instruction: str,
    queue_output: Queue,
    arm: Any,
    save_path: str | None = None,
) -> tuple[DetectedBox | None, bool]:
    detections = _run_yolo_detections(frame)
    selected_dets = _select_yolo_detections(detections, instruction)
    show_yolo_detection(
        frame,
        detections,
        status_lines=[
            f"instruction: {instruction}",
            f"selected: {len(selected_dets)} / {len(detections)}",
        ],
        save_path=save_path,
    )
    offset = get_config_value("catch_offset")

    llm_detections: list[DetectedFromLLM] = []
    for index, det in enumerate(selected_dets, start=1):
        item = _yolo_tuple_to_detected_from_llm(det, frame)
        item.id = index
        llm_detections.append(item)

    box, caught, _sam_debug = _refine_and_catch_first(
        frame,
        llm_detections,
        arm=arm,
        offset=offset,
        instruction=instruction,
        queue_output=queue_output,
        save_path=save_path,
    )
    return box, caught


def grasp_by_instruction(
    frame: cv2.typing.MatLike,
    instruction: str,
    queue_output: Queue,
    arm: Any,
) -> dict[str, Any]:
    if arm is None:
        raise RuntimeError("arm instance is required for Robonix grasp")

    backend = _get_instruction_detect_backend()
    save_path = _get_save_path()

    try:
        _prepare_arm_for_grasp(arm)
        if backend == "yolo":
            box, caught = _grasp_by_instruction_yolo(
                frame, instruction, queue_output, arm, save_path
            )
        else:
            box, caught = _grasp_by_instruction_llm(
                frame, instruction, queue_output, arm, save_path
            )
    except Exception as exc:
        log.error("grasp_by_instruction 异常: %s", exc, exc_info=True)
        return {
            "status": "failed",
            "reason": str(exc),
            "instruction": instruction,
            "method": backend,
            "grasp_success": False,
        }

    if box is None:
        return {
            "status": "failed",
            "reason": "未检测到目标物体",
            "instruction": instruction,
            "method": backend,
            "grasp_success": False,
        }

    return {
        "status": "success" if caught else "failed",
        "target": box.class_name,
        "instruction": instruction,
        "method": backend,
        "grasp_success": caught,
        "grasped_count": 1 if caught else 0,
        "detected_count": 1,
        "reason": None if caught else "抓取失败",
    }


def place_by_instruction(
    instruction: str,
    arm: Any,
    *,
    class_name: str = "",
) -> dict[str, Any]:
    """放置夹爪中的物体。instruction 为放置位置（如「盒子」），class_name 可选 fallback。"""
    if arm is None:
        raise RuntimeError("arm instance is required for Robonix place")

    place_instruction = (instruction or "").strip()
    fallback_class_name = (class_name or "").strip()
    if not place_instruction and not fallback_class_name:
        return {
            "status": "failed",
            "reason": "instruction（放置位置）与 class_name 不能同时为空",
            "instruction": place_instruction,
            "place_success": False,
        }

    place_pos = get_config_value("place_pos", default={}, raise_if_missing=False)
    try:
        ok = _place_at(
            arm,
            instruction=place_instruction,
            class_name=fallback_class_name,
            place_pos=place_pos,
        )
    except Exception as exc:
        log.error("place_by_instruction 异常: %s", exc, exc_info=True)
        return {
            "status": "failed",
            "reason": str(exc),
            "instruction": place_instruction,
            "target": fallback_class_name,
            "place_success": False,
        }

    return {
        "status": "success" if ok else "failed",
        "target": fallback_class_name,
        "instruction": place_instruction,
        "place_success": ok,
        "reason": None if ok else "放置失败",
    }


def _get_save_path() -> str:
    save_flag = get_config_value("save_img", raise_if_missing=False)
    if save_flag:
        return "../../grasp_debug.jpg"
    return None

def catch_by_instruction(
    frame: cv2.typing.MatLike,
    instruction: str,
    queue_output: Queue,
    success_callback: Optional[Callable[[], None]] = None,
    arm: Any = None,
) -> dict[str, Any]:
    """检测并循环执行「抓取 + 放置」，直到没有可抓取目标。"""
    if arm is None:
        raise RuntimeError("arm instance is required for Robonix grasp")

    backend = _get_instruction_detect_backend()
    boxes: list[str] = []
    grasped_count = 0
    detected_count = 0

    while True:
        grasp_result = grasp_by_instruction(frame, instruction, queue_output, arm)
        if grasp_result.get("status") != "success" or not grasp_result.get("grasp_success"):
            if not boxes:
                return grasp_result
            break

        target = str(grasp_result.get("target", ""))
        boxes.append(target)
        detected_count += 1

        place_result = place_by_instruction("", arm, class_name=target)
        if place_result.get("place_success"):
            grasped_count += 1
            if success_callback:
                success_callback()
        else:
            return {
                "status": "failed",
                "reason": place_result.get("reason", "放置失败"),
                "instruction": instruction,
                "method": backend,
                "grasp_success": grasped_count > 0,
                "grasped_count": grasped_count,
                "detected_count": detected_count,
                "target": ", ".join(boxes),
            }

    success_count = grasped_count
    return {
        "status": "success" if success_count > 0 else "failed",
        "target": ", ".join(boxes),
        "instruction": instruction,
        "method": backend,
        "grasp_success": success_count > 0,
        "grasped_count": success_count,
        "detected_count": detected_count,
    }
