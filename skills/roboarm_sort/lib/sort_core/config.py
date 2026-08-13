"""roboarm_sort 独立配置加载。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONFIG: dict[str, Any] = {}
_CONFIG_PATH: Path | None = None


@dataclass(frozen=True)
class LetterSpec:
    letter: str
    grasp_instruction: str
    yolo_class: str | None = None


@dataclass(frozen=True)
class SlotSpec:
    place_instruction: str
    pos: tuple[float, float] | tuple[float, float, float] | None = None


def load_config(config_file: str) -> dict[str, Any]:
    path = Path(config_file)
    if not path.is_file():
        raise FileNotFoundError(f"配置文件未找到: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def init_config(config_file: str) -> dict[str, Any]:
    global CONFIG, _CONFIG_PATH
    _CONFIG_PATH = Path(config_file).resolve()
    CONFIG = load_config(str(_CONFIG_PATH))
    return CONFIG


def get_config_value(
    key: str,
    default: Any = None,
    *,
    raise_if_missing: bool = True,
) -> Any:
    if raise_if_missing and key not in CONFIG:
        raise KeyError(f"配置项未找到: {key}")
    return CONFIG.get(key, default)


def get_allowed_letters() -> set[str]:
    letters = get_config_value("allowed_letters", ["C", "O", "S"])
    return {str(ch).upper() for ch in letters}


def get_slots() -> list[SlotSpec]:
    raw_slots = get_config_value("slots", raise_if_missing=False) or []
    slots: list[SlotSpec] = []
    for index, item in enumerate(raw_slots):
        if isinstance(item, str):
            slots.append(SlotSpec(place_instruction=item))
            continue
        if not isinstance(item, dict):
            raise ValueError(f"slots[{index}] 格式无效")
        place_instruction = str(
            item.get("place_instruction") or f"sort_slot_{index}"
        ).strip()
        pos_raw = item.get("pos")
        pos = None
        if pos_raw is not None:
            pos = tuple(float(v) for v in pos_raw)
        slots.append(SlotSpec(place_instruction=place_instruction, pos=pos))
    return slots


def get_letter_specs() -> dict[str, LetterSpec]:
    raw = get_config_value("letters", raise_if_missing=False) or {}
    specs: dict[str, LetterSpec] = {}
    for letter, data in raw.items():
        key = str(letter).upper()
        if not isinstance(data, dict):
            raise ValueError(f"letters.{key} 必须为对象")
        grasp_instruction = str(data.get("grasp_instruction", "")).strip()
        if not grasp_instruction:
            raise ValueError(f"letters.{key}.grasp_instruction 不能为空")
        yolo_class_raw = data.get("yolo_class")
        yolo_class = str(yolo_class_raw).strip().lower() if yolo_class_raw else None
        specs[key] = LetterSpec(
            letter=key,
            grasp_instruction=grasp_instruction,
            yolo_class=yolo_class or None,
        )
    return specs


def parse_order(order: str) -> list[str]:
    allowed = get_allowed_letters()
    letter_specs = get_letter_specs()
    cleaned = (order or "").upper().replace(" ", "").replace(",", "")
    if not cleaned:
        raise ValueError("order 不能为空")

    result: list[str] = []
    for ch in cleaned:
        if ch not in allowed:
            raise ValueError(
                f"order 含无效字母: {ch}，允许: {sorted(allowed)}"
            )
        if ch not in letter_specs:
            raise ValueError(f"letters 未配置字母: {ch}")
        result.append(ch)

    slots = get_slots()
    if len(result) > len(slots):
        raise ValueError(
            f"order 长度 {len(result)} 超过槽位数 {len(slots)}"
        )
    return result
