#!/usr/bin/env python3
"""离线 classify and grasp：直连相机 + LeRobot 臂，不启动 Robonix。

实时显示 YOLO 检测画面；按 g/空格 对当前检测结果执行抓取（与 skill 逻辑一致）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.bootstrap import setup

setup()

from lib.config import init_config
from lib.cv2_display import destroy_all_windows, poll_key, show_image
from lib.grasp_pipeline import (
    detect_all,
    draw_detections,
    grasp_detections,
    load_yolo_models,
    move_gripper_aside,
)
from lib.lerobo_arm import LeroboArm
from lib.local_camera import LocalCamera


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="离线 YOLO 分类抓取（不经过 Robonix）",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("TOOLS_CONFIG", ""),
        help="tools 配置文件路径（默认 tools/config.yaml）",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="检测并抓取一轮后退出（等同 skill classify_and_grasp）",
    )
    return parser.parse_args()


def _reinit_config(config_path: str) -> None:
    if config_path:
        init_config(config_path)


def _grab_color(camera: LocalCamera):
    frames = camera.get_frames()
    return frames.get("color")


def _run_grasp(
    arm: LeroboArm,
    frame,
    detections,
    *,
    window_name: str,
) -> tuple[int, int, list[str]]:
    progress_lines: list[str] = ["Grasping..."]

    def on_progress(msg: str) -> None:
        progress_lines.append(msg)
        print(msg)
        show_image(
            window_name,
            draw_detections(frame, detections, status_lines=progress_lines[-6:]),
        )

    show_image(
        window_name,
        draw_detections(
            frame,
            detections,
            status_lines=["Grasping...", "Press Esc to abort after current object"],
        ),
    )
    return grasp_detections(arm, detections, on_progress=on_progress)


def main() -> None:
    args = _parse_args()
    if args.config:
        _reinit_config(args.config)

    window_name = "Classify and Grasp"
    arm = LeroboArm()
    camera = LocalCamera(color=True, depth=False)
    models = load_yolo_models()

    try:
        move_gripper_aside(arm)
        arm.move_to_home(gripper_open_0to1=1)

        while True:
            frame = _grab_color(camera)
            if frame is None:
                print("Failed to grab frame")
                time.sleep(0.1)
                continue

            t0 = time.time()
            detections = detect_all(models, frame)
            fps = 1.0 / max(time.time() - t0, 1e-6)

            if args.once:
                annotated = draw_detections(
                    frame,
                    detections,
                    status_lines=[
                        f"Detections: {len(detections)}",
                        f"FPS: {fps:.1f}",
                        "Running grasp...",
                    ],
                )
                show_image(window_name, annotated)
                if not detections:
                    print("未检测到目标")
                else:
                    ok_n, fail_n, details = _run_grasp(
                        arm, frame, detections, window_name=window_name
                    )
                    summary = (
                        f"检测 {len(detections)} 个，成功 {ok_n}，失败 {fail_n}; "
                        + "; ".join(details)
                    )
                    print(summary)
                    show_image(
                        window_name,
                        draw_detections(
                            frame,
                            detections,
                            status_lines=[summary],
                        ),
                    )
                    poll_key(2000)
                break

            help_lines = [
                f"Detections: {len(detections)}  FPS: {fps:.1f}",
                "g/Space: grasp  h: home  Esc: quit",
            ]
            show_image(
                window_name,
                draw_detections(frame, detections, status_lines=help_lines),
            )

            key = poll_key(30) & 0xFF
            if key in (ord("g"), ord(" ")):
                if not detections:
                    print("未检测到目标，跳过抓取")
                    continue
                ok_n, fail_n, details = _run_grasp(
                    arm, frame, detections, window_name=window_name
                )
                summary = (
                    f"Done: ok={ok_n} fail={fail_n}; " + "; ".join(details)
                )
                print(summary)
                show_image(
                    window_name,
                    draw_detections(frame, detections, status_lines=[summary]),
                )
                poll_key(1500)
                move_gripper_aside(arm)
            elif key == ord("h"):
                arm.move_to_home(gripper_open_0to1=1)
            elif key == 27:
                break
    finally:
        try:
            arm.move_to_home(gripper_open_0to1=1)
        except Exception:
            pass
        arm.disconnect_arm()
        camera.close()
        destroy_all_windows()


if __name__ == "__main__":
    main()
