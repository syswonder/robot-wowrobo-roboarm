#!/usr/bin/env python3
"""orbbec_camera_roboarm — Robonix primitive for Orbbec RGB-D cameras."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Transform, TransformStamped
from robonix_api import Err, Ok, Primitive
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

log = logging.getLogger("orbbec_camera_roboarm")

try:
    import camera_pb2
except ImportError:  # pragma: no cover - available after rbnx codegen on target VM
    camera_pb2 = None  # type: ignore[assignment]

try:
    from pyorbbecsdk import Config, FrameSet, OBSensorType, Pipeline

    ORBBEC_SDK_AVAILABLE = True
except ImportError:
    ORBBEC_SDK_AVAILABLE = False
    Pipeline = None  # type: ignore[misc, assignment]


@dataclass
class FrameBundle:
    color_bgr: np.ndarray | None = None
    depth_u16: np.ndarray | None = None
    depth_colormap_bgr: np.ndarray | None = None


provider = Primitive(id="orbbec_camera_roboarm", namespace="robonix/primitive/camera")

_camera: "OrbbecCamera | None" = None
_stop = threading.Event()
_pub_thread: threading.Thread | None = None
_cfg: dict[str, Any] = {}
_latest_rgb: Image | None = None
_latest_depth: Image | None = None
_frame_lock = threading.Lock()


def _make_header(frame_id: str) -> Header:
    now = time.time()
    sec = int(now)
    nanosec = int((now - sec) * 1e9)
    return Header(stamp=Time(sec=sec, nanosec=nanosec), frame_id=frame_id)


def _bgr_to_rgb_image(array_bgr: np.ndarray, frame_id: str) -> Image:
    rgb = cv2.cvtColor(array_bgr, cv2.COLOR_BGR2RGB)
    msg = Image()
    msg.header = _make_header(frame_id)
    msg.height = int(rgb.shape[0])
    msg.width = int(rgb.shape[1])
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = int(rgb.shape[1] * 3)
    msg.data = rgb.tobytes()
    return msg


def _depth_u16_to_image(depth_u16: np.ndarray, frame_id: str) -> Image:
    msg = Image()
    msg.header = _make_header(frame_id)
    msg.height = int(depth_u16.shape[0])
    msg.width = int(depth_u16.shape[1])
    msg.encoding = "16UC1"
    msg.is_bigendian = 0
    msg.step = int(depth_u16.shape[1] * 2)
    msg.data = depth_u16.astype(np.uint16).tobytes()
    return msg


def _bgr_to_rgb8_image(array_bgr: np.ndarray, frame_id: str) -> Image:
    return _bgr_to_rgb_image(array_bgr, frame_id)


def _copy_ros_image_to_proto(ros_msg: Image):
    if camera_pb2 is None:
        raise RuntimeError("camera_pb2 is not generated; run rbnx build first")
    pb_image = camera_pb2.Image()
    pb_image.header.stamp.sec = ros_msg.header.stamp.sec
    pb_image.header.stamp.nanosec = ros_msg.header.stamp.nanosec
    pb_image.header.frame_id = ros_msg.header.frame_id
    pb_image.height = ros_msg.height
    pb_image.width = ros_msg.width
    pb_image.encoding = ros_msg.encoding
    pb_image.is_bigendian = ros_msg.is_bigendian
    pb_image.step = ros_msg.step
    pb_image.data = ros_msg.data
    return pb_image


class OrbbecCamera:
    """Local pyorbbecsdk or remote HTTP MJPEG camera backend."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._lock = threading.RLock()
        self._pipeline = None
        self._cap_rgb = None
        self._cap_depth = None
        self._frame_utils = None
        self._camera_info: CameraInfo | None = None

    @property
    def is_remote(self) -> bool:
        return bool(str(self._cfg.get("camera_ip", "")).strip())

    def connect(self) -> None:
        if self.is_remote:
            self._connect_remote()
        else:
            self._connect_local()
        self._camera_info = self._build_camera_info()

    def close(self) -> None:
        with self._lock:
            if self._pipeline is not None:
                try:
                    self._pipeline.stop()
                except Exception as exc:
                    log.warning("Failed to stop Orbbec pipeline: %s", exc)
                self._pipeline = None
            if self._cap_rgb is not None:
                self._cap_rgb.release()
                self._cap_rgb = None
            if self._cap_depth is not None:
                self._cap_depth.release()
                self._cap_depth = None

    def read(self) -> FrameBundle:
        with self._lock:
            if self.is_remote:
                return self._read_remote()
            return self._read_local()

    def camera_info(self) -> CameraInfo:
        if self._camera_info is None:
            self._camera_info = self._build_camera_info()
        msg = CameraInfo()
        msg.header = self._camera_info.header
        msg.height = self._camera_info.height
        msg.width = self._camera_info.width
        msg.distortion_model = self._camera_info.distortion_model
        msg.d = list(self._camera_info.d)
        msg.k = list(self._camera_info.k)
        msg.r = list(self._camera_info.r)
        msg.p = list(self._camera_info.p)
        return msg

    def extrinsics(self) -> TransformStamped:
        base_frame = str(self._cfg.get("base_frame", "base_link"))
        optical_frame = str(self._cfg.get("optical_frame", "camera_color_optical_frame"))
        translation = self._cfg.get("extrinsics_translation", [0.0, 0.0, 0.0])
        rotation_xyzw = self._cfg.get("extrinsics_rotation_xyzw", [0.0, 0.0, 0.0, 1.0])
        msg = TransformStamped()
        msg.header = _make_header(base_frame)
        msg.child_frame_id = optical_frame
        msg.transform = Transform(
            translation=_vec3(translation),
            rotation=_quat_xyzw(rotation_xyzw),
        )
        return msg

    def _connect_remote(self) -> None:
        host = str(self._cfg["camera_ip"])
        port = int(self._cfg["camera_port"])
        if self._cfg.get("enable_color", True):
            self._cap_rgb = cv2.VideoCapture(f"http://{host}:{port}/rgb_stream")
            if not self._cap_rgb.isOpened():
                raise ConnectionError(f"Failed to open RGB stream at {host}:{port}")
        if self._cfg.get("enable_depth", True):
            self._cap_depth = cv2.VideoCapture(f"http://{host}:{port}/depth_stream")
            if not self._cap_depth.isOpened():
                raise ConnectionError(f"Failed to open depth stream at {host}:{port}")
        log.info("Orbbec remote camera connected at %s:%s", host, port)

    def _connect_local(self) -> None:
        if not ORBBEC_SDK_AVAILABLE:
            raise RuntimeError(
                "pyorbbecsdk is not installed. Install it in the runtime venv."
            )
        import camera_utils as frame_utils

        self._frame_utils = frame_utils
        config = Config()
        pipeline = Pipeline()
        device = pipeline.get_device()
        video_sensors: list = []
        if self._cfg.get("enable_color", True):
            video_sensors.append(OBSensorType.COLOR_SENSOR)
        if self._cfg.get("enable_depth", True):
            video_sensors.append(OBSensorType.DEPTH_SENSOR)
        if not video_sensors:
            raise ValueError("At least one of enable_color or enable_depth must be true")

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
        log.info("Orbbec local camera connected via pyorbbecsdk")

    def _read_remote(self) -> FrameBundle:
        bundle = FrameBundle()
        if self._cap_rgb is not None:
            ok, frame = self._cap_rgb.read()
            if ok:
                bundle.color_bgr = frame
        if self._cap_depth is not None:
            ok, frame = self._cap_depth.read()
            if ok:
                bundle.depth_colormap_bgr = frame
        return bundle

    def _read_local(self) -> FrameBundle:
        if self._pipeline is None or self._frame_utils is None:
            raise RuntimeError("Orbbec local camera is not connected")
        frames: FrameSet = self._pipeline.wait_for_frames(
            int(self._cfg.get("frame_timeout_ms", 100))
        )
        bundle = FrameBundle()
        if frames is None:
            return bundle

        color_frame = frames.get_color_frame()
        if color_frame is not None:
            bundle.color_bgr = self._frame_utils.frame_to_bgr_image(color_frame)

        depth_frame = frames.get_depth_frame()
        if depth_frame is not None:
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape(depth_frame.get_height(), depth_frame.get_width())
            bundle.depth_u16 = depth_data
            depth_vis = np.zeros_like(depth_data, dtype=np.uint8)
            cv2.normalize(depth_data, depth_vis, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            bundle.depth_colormap_bgr = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        return bundle

    def _build_camera_info(self) -> CameraInfo:
        width = int(self._cfg.get("camera_width", 640))
        height = int(self._cfg.get("camera_height", 480))
        fx = float(self._cfg.get("fx", 600.0))
        fy = float(self._cfg.get("fy", 600.0))
        cx = float(self._cfg.get("cx", width / 2.0))
        cy = float(self._cfg.get("cy", height / 2.0))

        if not self.is_remote and self._pipeline is not None:
            try:
                camera_param = self._pipeline.get_camera_param()
                intrinsic = camera_param.rgb_intrinsic
                width = int(intrinsic.width)
                height = int(intrinsic.height)
                fx = float(intrinsic.fx)
                fy = float(intrinsic.fy)
                cx = float(intrinsic.cx)
                cy = float(intrinsic.cy)
            except Exception as exc:
                log.warning("Failed to read Orbbec camera intrinsics, using config: %s", exc)

        info = CameraInfo()
        info.header = _make_header(str(self._cfg.get("optical_frame", "camera_color_optical_frame")))
        info.width = width
        info.height = height
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info


def _vec3(values: list[float]):
    from geometry_msgs.msg import Vector3

    return Vector3(x=float(values[0]), y=float(values[1]), z=float(values[2]))


def _quat_xyzw(values: list[float]):
    from geometry_msgs.msg import Quaternion

    return Quaternion(
        x=float(values[0]),
        y=float(values[1]),
        z=float(values[2]),
        w=float(values[3]),
    )


def _build_driver_cfg(cfg: dict) -> dict[str, Any]:
    return {
        "camera_ip": str(cfg.get("camera_ip", "")).strip(),
        "camera_port": int(cfg.get("camera_port", 8083)),
        "enable_color": bool(cfg.get("enable_color", True)),
        "enable_depth": bool(cfg.get("enable_depth", True)),
        "publish_rate_hz": float(cfg.get("publish_rate_hz", 15)),
        "intrinsics_rate_hz": float(cfg.get("intrinsics_rate_hz", 1.0)),
        "rgb_topic": cfg.get("rgb_topic", "/roboarm/camera/rgb"),
        "depth_topic": cfg.get("depth_topic", "/roboarm/camera/depth"),
        "intrinsics_topic": cfg.get("intrinsics_topic", "/roboarm/camera/camera_info"),
        "extrinsics_topic": cfg.get("extrinsics_topic", "/roboarm/camera/extrinsics"),
        "optical_frame": cfg.get("optical_frame", "camera_color_optical_frame"),
        "base_frame": cfg.get("base_frame", "base_link"),
        "frame_timeout_ms": int(cfg.get("frame_timeout_ms", 100)),
        "camera_width": int(cfg.get("camera_width", 640)),
        "camera_height": int(cfg.get("camera_height", 480)),
        "fx": float(cfg.get("fx", 600.0)),
        "fy": float(cfg.get("fy", 600.0)),
        "cx": float(cfg.get("cx", 640.0 / 2.0)),
        "cy": float(cfg.get("cy", 480.0 / 2.0)),
        "extrinsics_translation": list(
            cfg.get("extrinsics_translation", [0.0, 0.0, 0.0])
        ),
        "extrinsics_rotation_xyzw": list(
            cfg.get("extrinsics_rotation_xyzw", [0.0, 0.0, 0.0, 1.0])
        ),
        "remote_depth_as_rgb": bool(cfg.get("remote_depth_as_rgb", True)),
        "allow_missing_hardware": bool(cfg.get("allow_missing_hardware", False)),
    }


@provider.on_init
def init(cfg: dict):
    global _camera, _cfg

    try:
        _cfg = _build_driver_cfg(cfg)
    except ValueError as exc:
        return Err(str(exc))

    provider.declare_ros2_topic(
        "robonix/primitive/camera/rgb",
        _cfg["rgb_topic"],
        qos="reliable",
    )
    provider.declare_ros2_topic(
        "robonix/primitive/camera/depth",
        _cfg["depth_topic"],
        qos="reliable",
    )
    provider.declare_ros2_topic(
        "robonix/primitive/camera/intrinsics",
        _cfg["intrinsics_topic"],
        qos="reliable",
    )
    provider.declare_ros2_topic(
        "robonix/primitive/camera/extrinsics",
        _cfg["extrinsics_topic"],
        qos="reliable",
    )

    provider.create_publisher(
        "robonix/primitive/camera/rgb",
        topic=_cfg["rgb_topic"],
        msg_type=Image,
        qos="reliable",
        declare=False,
    )
    provider.create_publisher(
        "robonix/primitive/camera/depth",
        topic=_cfg["depth_topic"],
        msg_type=Image,
        qos="reliable",
        declare=False,
    )
    provider.create_publisher(
        "robonix/primitive/camera/intrinsics",
        topic=_cfg["intrinsics_topic"],
        msg_type=CameraInfo,
        qos="reliable",
        declare=False,
    )
    provider.create_publisher(
        "robonix/primitive/camera/extrinsics",
        topic=_cfg["extrinsics_topic"],
        msg_type=TransformStamped,
        qos="reliable",
        declare=False,
    )

    log.info("orbbec_camera_roboarm initialized (hardware connect deferred to activate)")
    return Ok()


@provider.on_activate
def activate():
    global _pub_thread, _camera

    if _camera is None:
        try:
            _camera = OrbbecCamera(_cfg)
            _camera.connect()
        except Exception as exc:
            _camera = None
            target = (
                f"{_cfg['camera_ip']}:{_cfg['camera_port']}"
                if _cfg.get("camera_ip")
                else "local USB"
            )
            if _cfg.get("allow_missing_hardware", False):
                log.warning(
                    "Orbbec camera unavailable (%s): %s — running in degraded mode",
                    target,
                    exc,
                )
                log.info("orbbec_camera_roboarm activated (degraded, no frames)")
                return Ok()
            return Err(f"Failed to connect Orbbec camera ({target}): {exc}")

    _stop.clear()
    _pub_thread = threading.Thread(
        target=_publish_loop, name="orbbec_camera_pub", daemon=True
    )
    _pub_thread.start()
    log.info("orbbec_camera_roboarm activated")
    return Ok()


@provider.on_deactivate
def deactivate():
    _stop.set()
    if _pub_thread is not None and _pub_thread.is_alive():
        _pub_thread.join(timeout=2.0)
    log.info("orbbec_camera_roboarm deactivated")
    return Ok()


@provider.on_shutdown
def shutdown():
    global _camera

    deactivate()
    if _camera is not None:
        _camera.close()
        _camera = None
    log.info("orbbec_camera_roboarm shutdown complete")
    return Ok()


def _publish_loop() -> None:
    global _latest_rgb, _latest_depth

    rate_hz = max(1.0, float(_cfg.get("publish_rate_hz", 15)))
    period_s = 1.0 / rate_hz
    intrinsics_period_s = 1.0 / max(0.1, float(_cfg.get("intrinsics_rate_hz", 1.0)))
    next_intrinsics_at = 0.0
    optical_frame = str(_cfg.get("optical_frame", "camera_color_optical_frame"))
    depth_frame = str(_cfg.get("depth_frame", optical_frame))

    while not _stop.is_set():
        loop_start = time.monotonic()
        if _camera is None:
            break
        try:
            bundle = _camera.read()
            if bundle.color_bgr is not None:
                rgb_msg = _bgr_to_rgb_image(bundle.color_bgr, optical_frame)
                provider.emit("robonix/primitive/camera/rgb", rgb_msg)
                with _frame_lock:
                    _latest_rgb = rgb_msg

            if bundle.depth_u16 is not None:
                depth_msg = _depth_u16_to_image(bundle.depth_u16, depth_frame)
                provider.emit("robonix/primitive/camera/depth", depth_msg)
                with _frame_lock:
                    _latest_depth = depth_msg
            elif bundle.depth_colormap_bgr is not None and _cfg.get(
                "remote_depth_as_rgb", True
            ):
                depth_msg = _bgr_to_rgb8_image(bundle.depth_colormap_bgr, depth_frame)
                provider.emit("robonix/primitive/camera/depth", depth_msg)
                with _frame_lock:
                    _latest_depth = depth_msg

            if loop_start >= next_intrinsics_at:
                provider.emit(
                    "robonix/primitive/camera/intrinsics",
                    _camera.camera_info(),
                )
                provider.emit(
                    "robonix/primitive/camera/extrinsics",
                    _camera.extrinsics(),
                )
                next_intrinsics_at = loop_start + intrinsics_period_s
        except Exception as exc:
            log.warning("Failed to publish camera frames: %s", exc)

        elapsed = time.monotonic() - loop_start
        sleep_s = period_s - elapsed
        if sleep_s > 0:
            time.sleep(sleep_s)


def _capture_rgb_snapshot() -> Image:
    if _camera is None:
        raise RuntimeError("camera is not connected")
    bundle = _camera.read()
    if bundle.color_bgr is None:
        with _frame_lock:
            if _latest_rgb is not None:
                return _latest_rgb
        raise RuntimeError("failed to capture RGB frame")
    return _bgr_to_rgb_image(
        bundle.color_bgr,
        str(_cfg.get("optical_frame", "camera_color_optical_frame")),
    )


def _capture_depth_snapshot() -> Image:
    if _camera is None:
        raise RuntimeError("camera is not connected")
    bundle = _camera.read()
    depth_frame = str(_cfg.get("depth_frame", _cfg.get("optical_frame", "camera_color_optical_frame")))
    if bundle.depth_u16 is not None:
        return _depth_u16_to_image(bundle.depth_u16, depth_frame)
    if bundle.depth_colormap_bgr is not None:
        return _bgr_to_rgb8_image(bundle.depth_colormap_bgr, depth_frame)
    with _frame_lock:
        if _latest_depth is not None:
            return _latest_depth
    raise RuntimeError("failed to capture depth frame")


if camera_pb2 is not None:

    @provider.grpc("robonix/primitive/camera/snapshot")
    def snapshot(_request, _context=None):
        ros_image = _capture_rgb_snapshot()
        return camera_pb2.GetCameraImage_Response(
            image=_copy_ros_image_to_proto(ros_image)
        )

    @provider.grpc("robonix/primitive/camera/depth_snapshot")
    def depth_snapshot(_request, _context=None):
        ros_image = _capture_depth_snapshot()
        return camera_pb2.GetCameraImage_Response(
            image=_copy_ros_image_to_proto(ros_image)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    provider.run()
