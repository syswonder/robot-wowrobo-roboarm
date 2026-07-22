#!/usr/bin/env python3
"""测试 Orbbec 相机彩色 / 深度流是否正常。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.bootstrap import setup

setup()

import cv2
import numpy as np
from lib.cv2_display import destroy_all_windows, poll_key, show_image
from pyorbbecsdk import Config, FrameSet, OBSensorType, Pipeline

import camera_utils


def open_camera(color: bool, depth: bool) -> Pipeline:
    if not color and not depth:
        raise ValueError("At least one of color or depth must be True")
    config = Config()
    pipeline = Pipeline()
    device = pipeline.get_device()
    video_sensors = []
    if color:
        video_sensors.append(OBSensorType.COLOR_SENSOR)
    if depth:
        video_sensors.append(OBSensorType.DEPTH_SENSOR)
    sensor_list = device.get_sensor_list()
    for sensor in range(len(sensor_list)):
        try:
            sensor_type = sensor_list[sensor].get_type()
            if sensor_type in video_sensors:
                config.enable_stream(sensor_type)
        except Exception:
            continue
    pipeline.start(config)
    return pipeline


def close_camera(pipeline: Pipeline | None) -> None:
    if pipeline is not None:
        pipeline.stop()


def _process_color(frame):
    return camera_utils.frame_to_bgr_image(frame) if frame else None


def _process_depth(frame):
    if not frame:
        return None
    try:
        depth_data = np.frombuffer(frame.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape(frame.get_height(), frame.get_width())
        depth_image = np.zeros_like(depth_data, dtype=np.uint8)
        cv2.normalize(depth_data, depth_image, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
    except ValueError:
        return None


def get_frames(pipeline: Pipeline) -> dict[str, np.ndarray | None]:
    frames: FrameSet = pipeline.wait_for_frames(100)
    if frames is None:
        return {"color": None, "depth": None}
    return {
        "color": _process_color(frames.get_color_frame()),
        "depth": _process_depth(frames.get_depth_frame()),
    }


def main() -> None:
    pipeline = open_camera(True, True)
    try:
        while True:
            try:
                frames = get_frames(pipeline)
                color_image = frames.get("color")
                if color_image is None:
                    print("failed to get color image")
                else:
                    show_image("Color Viewer", color_image)
                depth_image = frames.get("depth")
                if depth_image is None:
                    print("failed to get depth image")
                else:
                    show_image("Depth Viewer", depth_image)
                key = poll_key(1)
                if key in (ord("q"), 27):
                    break
            except KeyboardInterrupt:
                break
    finally:
        destroy_all_windows()
        close_camera(pipeline)


if __name__ == "__main__":
    main()
