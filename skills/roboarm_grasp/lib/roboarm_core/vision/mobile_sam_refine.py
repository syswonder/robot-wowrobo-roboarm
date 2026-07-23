"""用 MobileSAM 根据 LLM 坐标点分割物体，再由最小外接矩形估计抓取位姿。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from pydantic import TypeAdapter

from roboarm_core.config import get_config_value, resolve_asset
from roboarm_core.llm.dataclass import DetectedBox, DetectedFromLLM, normalize_box_rotation_deg
from roboarm_core.llm.llm_api import extract_json_from_markdown

log = logging.getLogger("roboarm_grasp.mobile_sam")

_predictor: Any | None = None

MinAreaRect = tuple[tuple[float, float], tuple[float, float], float]


@dataclass
class SamRefineDebug:
    """dev 可视化：SAM 掩码与 minAreaRect。"""

    mask: np.ndarray | None = None
    min_area_rect: MinAreaRect | None = None
    prompt_point: tuple[int, int] | None = None
    llm_prompt_box: tuple[int, int, int, int] | None = None


@dataclass
class RefineResult:
    box: DetectedBox | None
    sam_debug: SamRefineDebug | None = None


def is_mobile_sam_enabled() -> bool:
    return bool(get_config_value("mobile_sam_enabled", True, raise_if_missing=False))


def _resolve_device() -> str:
    device_cfg = str(
        get_config_value("mobile_sam_device", "auto", raise_if_missing=False) or "auto"
    ).lower()
    if device_cfg != "auto":
        return device_cfg
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception as exc:
        log.warning("检测 torch 设备失败，回退到 cpu: %s", exc)
        return "cpu"


def _get_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor

    try:
        from mobile_sam import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise ImportError(
            "未安装 mobile_sam，请执行: pip install git+https://github.com/ChaoningZhang/MobileSAM.git"
        ) from exc

    model_type = str(
        get_config_value("mobile_sam_model_type", "vit_t", raise_if_missing=False)
        or "vit_t"
    )
    checkpoint_rel = str(
        get_config_value(
            "mobile_sam_checkpoint",
            "models/mobile_sam/mobile_sam.pt",
            raise_if_missing=False,
        )
    )
    checkpoint = resolve_asset(checkpoint_rel)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"MobileSAM 权重未找到: {checkpoint} "
            f"(配置 mobile_sam_checkpoint={checkpoint_rel})"
        )

    device = _resolve_device()
    log.info(
        "加载 MobileSAM: model=%s checkpoint=%s device=%s",
        model_type,
        checkpoint,
        device,
    )
    mobile_sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    mobile_sam.to(device=device)
    mobile_sam.eval()
    _predictor = SamPredictor(mobile_sam)
    return _predictor


def _llm_point_to_pixel(
    detection: DetectedFromLLM,
    img_w: int,
    img_h: int,
) -> tuple[int, int]:
    cx = round(detection.box_center_x * img_w)
    cy = round(detection.box_center_y * img_h)
    if get_config_value("RotationCam2Arm", raise_if_missing=False):
        cx = img_w - cx
        cy = img_h - cy
    cx = int(np.clip(cx, 0, img_w - 1))
    cy = int(np.clip(cy, 0, img_h - 1))
    return cx, cy


def _llm_box_to_xyxy(
    detection: DetectedFromLLM,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    cx, cy = _llm_point_to_pixel(detection, img_w, img_h)
    half_w = max(4, round(detection.box_width * img_w / 2))
    half_h = max(4, round(detection.box_height * img_h / 2))
    x1 = int(np.clip(cx - half_w, 0, img_w - 1))
    y1 = int(np.clip(cy - half_h, 0, img_h - 1))
    x2 = int(np.clip(cx + half_w, 0, img_w - 1))
    y2 = int(np.clip(cy + half_h, 0, img_h - 1))
    if x2 <= x1:
        x2 = min(img_w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(img_h - 1, y1 + 1)
    return x1, y1, x2, y2


def _oriented_rect_from_mask(
    mask: np.ndarray,
) -> tuple[tuple[float, float, float, float, float], MinAreaRect] | None:
    min_area = int(get_config_value("mobile_sam_min_mask_area", 200, raise_if_missing=False))
    mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        log.warning("MobileSAM 掩码未找到轮廓")
        return None

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < min_area:
        log.warning("MobileSAM 掩码面积过小: %.1f < %d", area, min_area)
        return None

    raw_rect: MinAreaRect = cv2.minAreaRect(contour)
    (cx, cy), (width, height), angle = raw_rect
    if width < 1.0 or height < 1.0:
        log.warning("MobileSAM 最小外接矩形尺寸无效: w=%.2f h=%.2f", width, height)
        return None

    norm_w, norm_h, norm_angle = float(width), float(height), float(angle)
    if norm_w < norm_h:
        norm_w, norm_h = norm_h, norm_w
        norm_angle += 90.0
    norm_angle = normalize_box_rotation_deg(norm_angle)
    return (float(cx), float(cy), norm_w, norm_h, norm_angle), raw_rect


def _select_best_mask(masks: np.ndarray, scores: np.ndarray) -> np.ndarray | None:
    if masks is None or len(masks) == 0:
        return None
    best_idx = int(np.argmax(scores))
    return masks[best_idx]


def refine_detection_with_mobile_sam(
    frame: cv2.typing.MatLike,
    detection: DetectedFromLLM,
) -> RefineResult:
    img_h, img_w = frame.shape[:2]
    cx, cy = _llm_point_to_pixel(detection, img_w, img_h)
    llm_box = _llm_box_to_xyxy(detection, img_w, img_h)
    debug = SamRefineDebug(prompt_point=(cx, cy), llm_prompt_box=llm_box)

    try:
        predictor = _get_predictor()
    except Exception as exc:
        log.error("MobileSAM 初始化失败: %s", exc, exc_info=True)
        return RefineResult(box=None, sam_debug=debug)

    try:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        point_coords = np.array([[cx, cy]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
        predict_kwargs: dict[str, Any] = {
            "point_coords": point_coords,
            "point_labels": point_labels,
            "multimask_output": bool(
                get_config_value("mobile_sam_multimask_output", True, raise_if_missing=False)
            ),
        }
        if bool(get_config_value("mobile_sam_use_box_prompt", True, raise_if_missing=False)):
            x1, y1, x2, y2 = llm_box
            predict_kwargs["box"] = np.array([x1, y1, x2, y2], dtype=np.float32)[None, :]

        masks, scores, _ = predictor.predict(**predict_kwargs)
        mask = _select_best_mask(masks, scores)
        if mask is None:
            log.warning("MobileSAM 未返回有效掩码 (point=%s)", (cx, cy))
            return RefineResult(box=None, sam_debug=debug)

        debug.mask = mask.astype(bool)

        rect_result = _oriented_rect_from_mask(mask)
        if rect_result is None:
            return RefineResult(box=None, sam_debug=debug)

        (box_cx, box_cy, box_w, box_h, angle), raw_rect = rect_result
        debug.min_area_rect = raw_rect

        log.info(
            "MobileSAM 精化成功: class=%s center=(%.1f, %.1f) size=(%.1f, %.1f) angle=%.1f score=%.3f",
            detection.class_name,
            box_cx,
            box_cy,
            box_w,
            box_h,
            angle,
            float(np.max(scores)),
        )
        return RefineResult(
            box=DetectedBox(
                class_name=detection.class_name,
                box_center_x=box_cx,
                box_center_y=box_cy,
                box_width=box_w,
                box_height=box_h,
                box_rotation_deg=angle,
                confidence=float(np.max(scores)),
            ),
            sam_debug=debug,
        )
    except Exception as exc:
        log.error(
            "MobileSAM 分割失败 (class=%s point=(%d,%d)): %s",
            detection.class_name,
            cx,
            cy,
            exc,
            exc_info=True,
        )
        return RefineResult(box=None, sam_debug=debug)


def parse_llm_detection(json_str: str) -> DetectedFromLLM | None:
    try:
        raw = json.loads(extract_json_from_markdown(json_str))
    except Exception as exc:
        log.error("解析 LLM JSON 失败: %s", exc, exc_info=True)
        return None

    if isinstance(raw, dict) and raw.get("failed"):
        log.info(
            "LLM 返回 failed=true: %s",
            raw.get("thinking_process", "未找到目标"),
        )
        return None

    try:
        detection = TypeAdapter(DetectedFromLLM).validate_python(raw)
    except Exception as exc:
        log.error("校验 LLM 检测字段失败: %s", exc, exc_info=True)
        return None

    if not detection.is_valid():
        log.warning("LLM 检测参数无效: %s", detection)
        return None
    return detection


def refine_llm_json_to_box(
    frame: cv2.typing.MatLike,
    json_str: str,
    *,
    img_w: int | None = None,
    img_h: int | None = None,
) -> RefineResult:
    """LLM 坐标 + MobileSAM 分割 + 最小外接矩形；失败时回退到 LLM 轴对齐框。"""
    detection = parse_llm_detection(json_str)
    if detection is None:
        return RefineResult(box=None)

    if is_mobile_sam_enabled():
        result = refine_detection_with_mobile_sam(frame, detection)
        if result.box is not None:
            return result
        log.warning("MobileSAM 精化失败，回退到 LLM 边界框")

    img_w = img_w if img_w is not None else frame.shape[1]
    img_h = img_h if img_h is not None else frame.shape[0]
    try:
        return RefineResult(box=detection.to_detected_box(img_w, img_h))
    except Exception as exc:
        log.error("LLM 边界框转换失败: %s", exc, exc_info=True)
        return RefineResult(box=None)
