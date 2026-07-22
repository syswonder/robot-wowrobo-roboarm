"""dev 模式下物体检测可视化（YOLO / LLM）。"""

from __future__ import annotations

import cv2
import numpy as np

from roboarm_core.config import get_config_value
from roboarm_core.cv2_display import poll_key, show_image
from roboarm_core.llm.dataclass import DetectedBox
from roboarm_core.llm.llm_detect import draw_boxes_on_frame

YOLO_WINDOW = "YOLO Detection"
LLM_WINDOW = "LLM Detection"

YoloDetection = tuple[tuple[float, float, float, float, float], float, int, str]


def is_dev_mode() -> bool:
    return bool(get_config_value("dev", False, raise_if_missing=False))


def draw_yolo_detections(
    frame: np.ndarray,
    detections: list[YoloDetection],
    *,
    status_lines: list[str] | None = None,
) -> np.ndarray:
    annotated = frame.copy()
    for (u, v, w, h, r), score, _class_id, class_name in detections:
        angle_deg = float(np.rad2deg(r))
        box_points = cv2.boxPoints(((u, v), (w, h), angle_deg))
        box_points = np.int64(box_points)
        cv2.drawContours(annotated, [box_points], 0, (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{class_name}: {score:.2f}",
            (int(u - w / 2), int(v - h / 2) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    y = 24
    for line in status_lines or []:
        cv2.putText(
            annotated,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
        )
        y += 24
    return annotated


def show_yolo_detection(
    frame: np.ndarray,
    detections: list[YoloDetection],
    *,
    status_lines: list[str] | None = None,
) -> None:
    if not is_dev_mode():
        return
    annotated = draw_yolo_detections(frame, detections, status_lines=status_lines)
    show_image(YOLO_WINDOW, annotated)
    poll_key(1)


def show_llm_detection(
    frame: np.ndarray,
    boxes: list[DetectedBox],
    *,
    status_lines: list[str] | None = None,
) -> None:
    if not is_dev_mode():
        return
    annotated = draw_boxes_on_frame(boxes, frame) if boxes else frame.copy()
    y = 24
    for line in status_lines or []:
        cv2.putText(
            annotated,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
        )
        y += 24
    show_image(LLM_WINDOW, annotated)
    poll_key(1)
