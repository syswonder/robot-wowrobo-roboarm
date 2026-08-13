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
    """单次抓取：只返回得分最高的一个检测框。"""
    if not detections:
        return []
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
    return [scored[0][1]]


def _select_yolo_detections_all(
    detections: list[tuple],
    instruction: str,
) -> list[tuple]:
    """批量抓取：返回所有得分达标的检测框。"""
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
    return_home_after_catch: bool = True,
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
    if ok and return_home_after_catch:
        arm.move_to_home(gripper_open_0to1=0)
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
    if not get_config_value(
        "move_gripper_aside_for_camera", True, raise_if_missing=False
    ):
        return
    default_gripper_aside_pos = get_config_value(
        "default_gripper_aside_pos", raise_if_missing=False
    )
    if default_gripper_aside_pos is not None:
        arm.move_to(default_gripper_aside_pos, block_until_reach=True)
    time.sleep(0.5)


def _frame_after_prepare_arm(
    arm: Any,
    frame: cv2.typing.MatLike | None,
    get_frame: Callable[[], cv2.typing.MatLike | None] | None,
) -> cv2.typing.MatLike | None:
    _prepare_arm_for_grasp(arm)
    if frame is not None:
        return frame
    if get_frame is None:
        raise RuntimeError("frame 或 get_frame 必须提供其一")
    return get_frame()


def _refine_and_catch_first(
    frame: cv2.typing.MatLike,
    detections: list[DetectedFromLLM],
    *,
    arm: Any,
    offset: float,
    instruction: str,
    queue_output: Queue,
    save_path: str | None = None,
    return_home_after_catch: bool = True,
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
            arm,
            box,
            offset=offset,
            queue_output=queue_output,
            return_home_after_catch=return_home_after_catch,
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


def _yolo_selected_detections(
    frame: cv2.typing.MatLike,
    instruction: str,
    *,
    select_all: bool,
    save_path: str | None = None,
) -> tuple[list[tuple], list[tuple]]:
    detections = _run_yolo_detections(frame)
    if select_all:
        selected = _select_yolo_detections_all(detections, instruction)
        status = f"selected all: {len(selected)} / {len(detections)}"
    else:
        selected = _select_yolo_detections(detections, instruction)
        status = f"selected: {len(selected)} / {len(detections)}"
    show_yolo_detection(
        frame,
        detections,
        status_lines=[f"instruction: {instruction}", status],
        save_path=save_path,
    )
    return detections, selected


def _catch_yolo_dets_first(
    frame: cv2.typing.MatLike,
    detections: list[tuple],
    *,
    arm: Any,
    offset: float,
    instruction: str,
    queue_output: Queue,
    save_path: str | None = None,
    return_home_after_catch: bool = True,
) -> tuple[DetectedBox | None, bool]:
    if not detections:
        return None, False

    for det in detections:
        box = _yolo_tuple_to_box(det)
        ok, _, _, _ = _catch_box(
            arm,
            box,
            offset=offset,
            queue_output=queue_output,
            return_home_after_catch=return_home_after_catch,
        )
        if ok:
            show_yolo_detection(
                frame,
                detections,
                status_lines=[
                    f"instruction: {instruction}",
                    f"caught: {box.class_name}",
                ],
                save_path=save_path,
            )
            return box, True

    show_yolo_detection(
        frame,
        detections,
        status_lines=[
            f"instruction: {instruction}",
            "no successful catch",
        ],
        save_path=save_path,
    )
    return None, False


def _await_llm_detections(
    frame: cv2.typing.MatLike,
    instruction: str,
    *,
    prompt_key: str,
    save_path: str | None = None,
) -> list[DetectedFromLLM]:
    llm_detect = LLMDetect()
    response_task = llm_detect.detect_frame(
        frame,
        prompt_key=prompt_key,
        replace_map={"{user_instruction}": instruction},
        schema=TypeAdapter(InstructionDetectResponse).json_schema(),
    )
    detections: list[DetectedFromLLM] = []
    processed = False

    while True:
        response, done = llm_detect.llm_api.await_task(response_task, blocking=False)
        if response and not processed:
            detections = parse_llm_detections(response)
            processed = True
        elif not processed:
            show_llm_detection(
                frame,
                [],
                status_lines=["instruction: " + instruction, "waiting for LLM..."],
                save_path=save_path,
            )
        if done:
            break
    return detections


def detect_all_by_instruction(
    frame: cv2.typing.MatLike,
    instruction: str,
    *,
    save_path: str | None = None,
) -> list[DetectedFromLLM]:
    """使用 grasp_all prompt 返回所有符合指令的检测目标（仅 LLM 后端）。"""
    detections = _await_llm_detections(
        frame,
        instruction,
        prompt_key="grasp_all_instruction_prompt",
        save_path=save_path,
    )
    show_llm_detection(
        frame,
        [],
        status_lines=[
            "instruction: " + instruction,
            f"detected all: {len(detections)}",
        ],
        save_path=save_path,
    )
    return detections


def _grasp_by_instruction_llm(
    frame: cv2.typing.MatLike,
    instruction: str,
    queue_output: Queue,
    arm: Any,
    save_path: str | None = None,
) -> tuple[DetectedBox | None, bool]:
    detections = _await_llm_detections(
        frame,
        instruction,
        prompt_key="user_instruction_prompt",
        save_path=save_path,
    )
    if detections:
        detections = detections[:1]
    offset = get_config_value("catch_offset")
    return_home_after_catch = get_config_value(
        "return_home_after_grasp_by_instruction", True, raise_if_missing=False
    )
    if not detections:
        show_llm_detection(
            frame,
            [],
            status_lines=[
                "instruction: " + instruction,
                "no target detected",
            ],
            save_path=save_path,
        )
        return None, False

    box, caught, _sam_debug = _refine_and_catch_first(
        frame,
        detections,
        arm=arm,
        offset=offset,
        instruction=instruction,
        queue_output=queue_output,
        save_path=save_path,
        return_home_after_catch=return_home_after_catch,
    )
    return box, caught


def _grasp_by_instruction_yolo(
    frame: cv2.typing.MatLike,
    instruction: str,
    queue_output: Queue,
    arm: Any,
    save_path: str | None = None,
) -> tuple[DetectedBox | None, bool]:
    _detections, selected_dets = _yolo_selected_detections(
        frame, instruction, select_all=False, save_path=save_path
    )
    if not selected_dets:
        return None, False

    offset = get_config_value("catch_offset")
    return_home_after_catch = get_config_value(
        "return_home_after_grasp_by_instruction", True, raise_if_missing=False
    )
    return _catch_yolo_dets_first(
        frame,
        selected_dets,
        arm=arm,
        offset=offset,
        instruction=instruction,
        queue_output=queue_output,
        save_path=save_path,
        return_home_after_catch=return_home_after_catch,
    )


def grasp_by_instruction(
    frame: cv2.typing.MatLike | None,
    instruction: str,
    queue_output: Queue,
    arm: Any,
    *,
    get_frame: Callable[[], cv2.typing.MatLike | None] | None = None,
) -> dict[str, Any]:
    if arm is None:
        raise RuntimeError("arm instance is required for Robonix grasp")

    backend = _get_instruction_detect_backend()
    save_path = _get_save_path()

    try:
        frame = _frame_after_prepare_arm(arm, frame, get_frame)
        if frame is None:
            return {
                "status": "failed",
                "reason": "无法获取相机画面",
                "instruction": instruction,
                "method": backend,
                "grasp_success": False,
            }
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

def grasp_all_by_instruction(
    frame: cv2.typing.MatLike | None,
    instruction: str,
    queue_output: Queue,
    arm: Any,
    *,
    get_frame: Callable[[], cv2.typing.MatLike | None] | None = None,
) -> dict[str, Any]:
    """一次性检测所有符合指令的物体，再逐个 refine、抓取并放置。"""
    if arm is None:
        raise RuntimeError("arm instance is required for Robonix grasp")

    backend = _get_instruction_detect_backend()
    save_path = _get_save_path()
    boxes: list[str] = []
    grasped_count = 0
    detected_count = 0

    try:
        frame = _frame_after_prepare_arm(arm, frame, get_frame)
        if frame is None:
            return {
                "status": "failed",
                "reason": "无法获取相机画面",
                "instruction": instruction,
                "method": backend,
                "grasp_success": False,
            }
        offset = get_config_value("catch_offset")

        if backend == "yolo":
            _detections, selected_yolo = _yolo_selected_detections(
                frame, instruction, select_all=True, save_path=save_path
            )
            if not selected_yolo:
                return {
                    "status": "failed",
                    "reason": "未检测到目标物体",
                    "instruction": instruction,
                    "method": backend,
                    "grasp_success": False,
                }
            detected_count = len(selected_yolo)
            for det in selected_yolo:
                box, caught = _catch_yolo_dets_first(
                    frame,
                    [det],
                    arm=arm,
                    offset=offset,
                    instruction=instruction,
                    queue_output=queue_output,
                    save_path=save_path,
                )
                if not caught or box is None:
                    continue

                boxes.append(box.class_name)
                place_result = place_by_instruction("", arm, class_name=box.class_name)
                if place_result.get("place_success"):
                    grasped_count += 1
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
        else:
            detections = detect_all_by_instruction(
                frame, instruction, save_path=save_path
            )
            if not detections:
                return {
                    "status": "failed",
                    "reason": "未检测到目标物体",
                    "instruction": instruction,
                    "method": backend,
                    "grasp_success": False,
                }

            detected_count = len(detections)
            for detection in detections:
                box, caught, _sam_debug = _refine_and_catch_first(
                    frame,
                    [detection],
                    arm=arm,
                    offset=offset,
                    instruction=instruction,
                    queue_output=queue_output,
                    save_path=save_path,
                )
                if not caught or box is None:
                    continue

                boxes.append(box.class_name)
                place_result = place_by_instruction("", arm, class_name=box.class_name)
                if place_result.get("place_success"):
                    grasped_count += 1
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
    except Exception as exc:
        log.error("grasp_all_by_instruction 异常: %s", exc, exc_info=True)
        return {
            "status": "failed",
            "reason": str(exc),
            "instruction": instruction,
            "method": backend,
            "grasp_success": grasped_count > 0,
            "grasped_count": grasped_count,
            "detected_count": detected_count,
            "target": ", ".join(boxes),
        }

    return {
        "status": "success" if grasped_count > 0 else "failed",
        "target": ", ".join(boxes),
        "instruction": instruction,
        "method": backend,
        "grasp_success": grasped_count > 0,
        "grasped_count": grasped_count,
        "detected_count": detected_count,
        "reason": None if grasped_count > 0 else "抓取失败",
    }


def catch_by_instruction(
    frame: cv2.typing.MatLike,
    instruction: str,
    queue_output: Queue,
    success_callback: Optional[Callable[[], None]] = None,
    arm: Any = None,
) -> dict[str, Any]:
    """检测并循环执行「抓取 + 放置」，直到没有可抓取目标。"""
    result = grasp_all_by_instruction(
        frame,
        instruction,
        queue_output,
        arm,
    )
    if success_callback and result.get("grasped_count", 0) > 0:
        for _ in range(int(result["grasped_count"])):
            success_callback()
    return result
