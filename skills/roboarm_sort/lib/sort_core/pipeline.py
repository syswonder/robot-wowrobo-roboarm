"""C/O/S 排序编排：解析 order，依次调用 grasp / place。"""

from __future__ import annotations

import logging
from typing import Any

from sort_core.config import get_letter_specs, get_slots, parse_order
from sort_core import grasp_client

log = logging.getLogger("roboarm_sort.pipeline")


def sort_letters(order: str) -> dict[str, Any]:
    sequence = parse_order(order)
    letter_specs = get_letter_specs()
    slots = get_slots()
    details: list[str] = []

    for index, letter in enumerate(sequence):
        spec = letter_specs[letter]
        slot = slots[index]
        pos_hint = ""
        if slot.pos is not None:
            pos_hint = f" place_ref={slot.pos}"

        log.info(
            "sort step %d/%d: letter=%s grasp=%r place_instruction=%r%s",
            index + 1,
            len(sequence),
            letter,
            spec.grasp_instruction,
            slot.place_instruction,
            pos_hint,
        )

        grasp_resp = grasp_client.grasp_by_instruction(spec.grasp_instruction)
        if not grasp_client.grasp_success(grasp_resp):
            msg = grasp_client.response_message(grasp_resp)
            detail = f"{letter}: 抓取失败 ({msg})"
            details.append(detail)
            return {
                "status": "failed",
                "reason": detail,
                "order": order,
                "completed_steps": index,
                "details": details,
            }

        class_name = grasp_client.grasp_class_name(grasp_resp)
        if spec.yolo_class and class_name.lower() != spec.yolo_class.lower():
            detail = (
                f"{letter}: 抓错物体 (期望 {spec.yolo_class}, 实际 {class_name or '无'})"
            )
            details.append(detail)
            return {
                "status": "failed",
                "reason": detail,
                "order": order,
                "completed_steps": index,
                "details": details,
            }

        place_resp = grasp_client.place_by_instruction(
            slot.place_instruction,
            class_name=class_name,
            pos=slot.pos,
        )
        if not grasp_client.place_success(place_resp):
            msg = grasp_client.response_message(place_resp)
            detail = f"{letter}: 放置失败 ({msg})"
            details.append(detail)
            return {
                "status": "failed",
                "reason": detail,
                "order": order,
                "completed_steps": index,
                "details": details,
            }

        details.append(f"{letter}: OK -> {slot.place_instruction}")

    return {
        "status": "success",
        "reason": None,
        "order": order,
        "completed_steps": len(sequence),
        "details": details,
    }
