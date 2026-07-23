"""放置位置解析：配置默认值、类别/关键词匹配、指令中的左/右修饰。"""

from __future__ import annotations

import copy
from typing import Any

from roboarm_core.config import get_config_value


def _resolve_pos_refs(
    pos_template: list[Any],
    *,
    target_x: float,
    target_y: float,
) -> list[float]:
    resolved = copy.deepcopy(pos_template)
    for index, ref in enumerate(resolved):
        if ref == "x":
            resolved[index] = target_x
        elif ref == "-x":
            resolved[index] = -target_x
        elif ref == "y":
            resolved[index] = target_y
        elif ref == "-y":
            resolved[index] = -target_y
        else:
            resolved[index] = float(ref)
    return [float(value) for value in resolved]


def _find_place_template(class_name: str, place_pos: dict[str, Any]) -> list[Any] | None:
    class_data = place_pos.get(class_name)
    if class_data is not None and "pos" in class_data:
        return class_data["pos"]

    lower_name = class_name.lower()
    for block_name, pos_data in place_pos.items():
        if block_name.startswith("_"):
            continue
        if block_name.lower() in lower_name:
            return pos_data.get("pos")
        for keyword in pos_data.get("keywords", []):
            if str(keyword).lower() in lower_name:
                return pos_data.get("pos")
    return None


def apply_instruction_x_sign(pos: list[float], instruction: str) -> list[float]:
    """根据指令中的「左」「右」调整放置点横坐标符号。"""
    if not instruction or len(pos) < 1:
        return pos
    has_left = "左" in instruction or "left" in instruction
    has_right = "右" in instruction or "right" in instruction
    if has_left and not has_right:
        pos[0] = -abs(pos[0])
    elif has_right and not has_left:
        pos[0] = abs(pos[0])
    return pos


def resolve_place_pos(
    class_name: str,
    *,
    instruction: str = "",
    place_pos: dict[str, Any] | None = None,
    target_x: float = 0.0,
    target_y: float = 0.0,
) -> list[float]:
    if place_pos is None:
        place_pos = get_config_value("place_pos", default={}, raise_if_missing=False)

    template = _find_place_template(class_name, place_pos)
    if template is None:
        default = get_config_value(
            "default_place_pos", default=[0.1, 0.1], raise_if_missing=False
        )
        template = default if default else [0.1, 0.1]

    resolved = _resolve_pos_refs(template, target_x=target_x, target_y=target_y)
    return apply_instruction_x_sign(resolved, instruction)
