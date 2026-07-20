#!/usr/bin/env python3
"""WASD 键盘控制机械臂末端位置，Esc 退出。"""

from __future__ import annotations

import os
import select
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.bootstrap import setup

setup()

from lib.lerobo_arm import LeroboArm

ROT_STEP_RAD = 0.1
TARGET_POSE = np.array([0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 1.0])


def handle_key(key: str) -> None:
    global TARGET_POSE
    if key == "w":
        TARGET_POSE[1] += 0.01
    elif key == "s":
        TARGET_POSE[1] -= 0.01
    elif key == "a":
        TARGET_POSE[0] -= 0.01
    elif key == "d":
        TARGET_POSE[0] += 0.01
    elif key == "z":
        TARGET_POSE[2] -= 0.01
    elif key == "x":
        TARGET_POSE[2] += 0.01
    elif key == "u":
        TARGET_POSE[3] += ROT_STEP_RAD
    elif key == "j":
        TARGET_POSE[3] -= ROT_STEP_RAD
    elif key == "e":
        TARGET_POSE[6] = 0.0
    elif key == "q":
        TARGET_POSE[6] = 1.0
    elif key == "r":
        TARGET_POSE[:3] = np.random.uniform([-0.2, 0.0, 0.07], [0.2, 0.3, 0.17])
    elif key == "\x1b":
        TARGET_POSE = np.empty(0)
        return
    print("Current pose:", TARGET_POSE.round(2))


def read_key(timeout: float = 0.1) -> str | None:
    if os.name == "nt":
        import msvcrt

        if not msvcrt.kbhit():
            time.sleep(timeout)
            return None
        key = msvcrt.getwch()
        if key in ("\x00", "\xe0"):
            key += msvcrt.getwch()
        return key

    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        readable, _, _ = select.select([sys.stdin], [], [], timeout)
        if not readable:
            return None
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def input_loop() -> None:
    while True:
        key = read_key()
        if key is None:
            continue
        handle_key(key)
        if len(TARGET_POSE) != 7:
            return


def main() -> None:
    global TARGET_POSE
    thread = threading.Thread(target=input_loop, daemon=True)
    thread.start()

    arm = LeroboArm()
    arm.move_to_home(gripper_open_0to1=1)
    pos, rot = arm.get_arm_pose()
    if pos:
        TARGET_POSE[:3] = np.array(pos)
    if rot:
        TARGET_POSE[3:6] = np.deg2rad(rot)

    while True:
        if len(TARGET_POSE) != 7:
            arm.move_to_home(gripper_open_0to1=1)
            arm.disconnect_arm()
            return
        try:
            arm.move_to(
                TARGET_POSE[:3].tolist(),
                gripper_open_0to1=float(TARGET_POSE[6]),
                rot_rad=float(TARGET_POSE[3]),
            )
        except Exception as exc:
            print("Error:", exc)
            time.sleep(0.5)


if __name__ == "__main__":
    main()
