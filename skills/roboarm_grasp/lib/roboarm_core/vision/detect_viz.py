"""dev 模式下物体检测可视化（YOLO / LLM / MobileSAM）。"""

from __future__ import annotations

import cv2
import numpy as np

from roboarm_core.config import get_config_value
from roboarm_core.cv2_display import poll_key, show_image
from roboarm_core.llm.dataclass import DetectedBox
from roboarm_core.llm.llm_detect import draw_boxes_on_frame
from roboarm_core.vision.mobile_sam_refine import SamRefineDebug

YOLO_WINDOW = "YOLO Detection"
LLM_WINDOW = "LLM Detection"

YoloDetection = tuple[tuple[float, float, float, float, float], float, int, str]


def is_dev_mode() -> bool:
    return bool(get_config_value("dev", False, raise_if_missing=False))


def draw_sam_debug_on_frame(
    frame: np.ndarray,
    sam_debug_list: list[SamRefineDebug],
) -> np.ndarray:
    annotated = frame.copy()
    for debug in sam_debug_list:
        if debug.mask is not None:
            overlay = annotated.copy()
            overlay[debug.mask] = (0, 255, 255)
            annotated = cv2.addWeighted(annotated, 0.15, overlay, 0.85, 0)

        if debug.min_area_rect is not None:
            box_points = cv2.boxPoints(debug.min_area_rect)
            box_points = np.int32(box_points)
            cv2.drawContours(annotated, [box_points], 0, (0, 255, 0), 2)
            cx, cy = debug.min_area_rect[0]
            cv2.putText(
                annotated,
                "minAreaRect",
                (int(cx), int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )

        # if debug.llm_prompt_box is not None:
        #     x1, y1, x2, y2 = debug.llm_prompt_box
        #     cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 128, 0), 1)

        if debug.prompt_point is not None:
            cv2.circle(annotated, debug.prompt_point, 6, (255, 0, 0), -1)
            cv2.putText(
                annotated,
                "LLM pt",
                (debug.prompt_point[0] + 8, debug.prompt_point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 0),
                1,
            )
    return annotated


def draw_llm_detection_frame(
    frame: np.ndarray,
    boxes: list[DetectedBox],
    *,
    sam_debug_list: list[SamRefineDebug] | None = None,
    status_lines: list[str] | None = None,
) -> np.ndarray:
    annotated = frame.copy()
    if sam_debug_list:
        annotated = draw_sam_debug_on_frame(annotated, sam_debug_list)
    if boxes:
        annotated = draw_boxes_on_frame(boxes, annotated)

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
    save_path: str | None = None,
) -> None:
    if not is_dev_mode():
        return
    annotated = draw_yolo_detections(frame, detections, status_lines=status_lines)
    if save_path:
        cv2.imwrite(save_path, annotated)
    show_image(YOLO_WINDOW, annotated)
    poll_key(1)


def show_llm_detection(
    frame: np.ndarray,
    boxes: list[DetectedBox],
    *,
    sam_debug_list: list[SamRefineDebug] | None = None,
    status_lines: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    if not is_dev_mode():
        return
    annotated = draw_llm_detection_frame(
        frame,
        boxes,
        sam_debug_list=sam_debug_list,
        status_lines=status_lines,
    )
    if save_path:
        cv2.imwrite(save_path, annotated)
    show_image(LLM_WINDOW, annotated)
    poll_key(1)
