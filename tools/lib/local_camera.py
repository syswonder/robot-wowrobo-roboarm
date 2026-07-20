"""Local Orbbec USB camera for offline tools."""

from __future__ import annotations

import cv2
import numpy as np
from lib.config import get_config_value

try:
    from pyorbbecsdk import Config, FrameSet, OBSensorType, Pipeline

    ORBBEC_SDK_AVAILABLE = True
except ImportError:
    ORBBEC_SDK_AVAILABLE = False
    Pipeline = None  # type: ignore[misc, assignment]

import camera_utils


class LocalCamera:
    def __init__(self, *, color: bool = True, depth: bool = False) -> None:
        self.color = color
        self.depth = depth
        self._pipeline = None
        self._cap_rgb = None
        self._cap_depth = None
        self._ip = str(get_config_value("camera_ip", "", raise_if_missing=False) or "").strip()
        self._port = int(get_config_value("camera_port", 8083, raise_if_missing=False))
        self._connect()

    def _connect(self) -> None:
        if self._ip:
            if self.color:
                self._cap_rgb = cv2.VideoCapture(
                    f"http://{self._ip}:{self._port}/rgb_stream"
                )
                if not self._cap_rgb.isOpened():
                    raise ConnectionError(
                        f"Failed to open RGB stream at {self._ip}:{self._port}"
                    )
            if self.depth:
                self._cap_depth = cv2.VideoCapture(
                    f"http://{self._ip}:{self._port}/depth_stream"
                )
            return

        if not ORBBEC_SDK_AVAILABLE:
            raise RuntimeError("pyorbbecsdk is required for local USB camera")
        config = Config()
        pipeline = Pipeline()
        device = pipeline.get_device()
        video_sensors: list = []
        if self.color:
            video_sensors.append(OBSensorType.COLOR_SENSOR)
        if self.depth:
            video_sensors.append(OBSensorType.DEPTH_SENSOR)
        sensor_list = device.get_sensor_list()
        for sensor_index in range(len(sensor_list)):
            try:
                sensor_type = sensor_list[sensor_index].get_type()
                if sensor_type in video_sensors:
                    config.enable_stream(sensor_type)
            except Exception:
                continue
        pipeline.start(config)
        self._pipeline = pipeline

    def get_frames(self) -> dict[str, np.ndarray | None]:
        if self._ip:
            frame_rgb = None
            frame_depth = None
            if self._cap_rgb is not None:
                ok, frame_rgb = self._cap_rgb.read()
                if not ok:
                    frame_rgb = None
            if self._cap_depth is not None:
                ok, frame_depth = self._cap_depth.read()
                if not ok:
                    frame_depth = None
            return {"color": frame_rgb, "depth": frame_depth}

        if self._pipeline is None:
            return {"color": None, "depth": None}
        frames: FrameSet = self._pipeline.wait_for_frames(100)
        if frames is None:
            return {"color": None, "depth": None}

        color_bgr = None
        depth_vis = None
        color_frame = frames.get_color_frame()
        if color_frame is not None:
            color_bgr = camera_utils.frame_to_bgr_image(color_frame)
        depth_frame = frames.get_depth_frame()
        if depth_frame is not None:
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape(
                depth_frame.get_height(), depth_frame.get_width()
            )
            depth_vis = np.zeros_like(depth_data, dtype=np.uint8)
            cv2.normalize(depth_data, depth_vis, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        return {"color": color_bgr, "depth": depth_vis}

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        if self._cap_rgb is not None:
            self._cap_rgb.release()
            self._cap_rgb = None
        if self._cap_depth is not None:
            self._cap_depth.release()
            self._cap_depth = None
