#!/usr/bin/env python3
"""YOLO OBB 实时检测预览（直连相机，不经过 Robonix）。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.bootstrap import setup

setup()

from lib.cv2_display import destroy_all_windows, poll_key, show_image
from lib.local_camera import LocalCamera
from lib.config import get_config_value, resolve_asset
from lib.yolo_detect import detect_objects_in_frame, load_model


def draw_box(frame, u, v, w, h, angle_deg, label):
    box_points = cv2.boxPoints(((u, v), (w, h), angle_deg))
    box_points = np.int64(box_points)
    cv2.drawContours(frame, [box_points], 0, (0, 255, 0), 2)
    cv2.putText(
        frame,
        label,
        (int(u - w / 2), int(v - h / 2) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )


def main() -> None:
    model_paths = [
        resolve_asset(path)
        for path in get_config_value("classification_YOLO_model_path", [])
    ]
    if not model_paths:
        raise RuntimeError("classification_YOLO_model_path 未配置")
    model = load_model(str(model_paths[0]))
    conf_thres = get_config_value("default_conf_thres")

    camera = LocalCamera(color=True, depth=False)
    try:
        while True:
            frames = camera.get_frames()
            color = frames.get("color")
            if color is None:
                print("Failed to grab frame")
                continue

            start = time.time()
            detections = detect_objects_in_frame(
                model, color, conf_thres=conf_thres
            )
            annotated = color.copy()
            for (x, y, w, h, r), score, _class_id, class_name in detections:
                draw_box(
                    annotated,
                    x,
                    y,
                    w,
                    h,
                    np.rad2deg(r),
                    f"{class_name}: {score:.2f}",
                )
            fps = 1.0 / max(time.time() - start, 1e-6)
            cv2.putText(
                annotated,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            show_image("YOLO Detection", annotated)
            if poll_key(1) & 0xFF == 27:
                break
    finally:
        camera.close()
        destroy_all_windows()


if __name__ == "__main__":
    main()
