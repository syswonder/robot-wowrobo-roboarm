#!/usr/bin/env python3
"""读取各关节原始角度，用于标定 arm_offset（真机专用）。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.bootstrap import setup

setup()

from lib.lerobo_arm import LeroboArm


def main() -> None:
    arm = LeroboArm()
    arm.disable_torque()
    print("将机械臂手动摆到零位，观察下方原始关节角（度），Ctrl+C 退出。")
    print("把读数写入 robonix_manifest.yaml 的 arm_offset。")
    try:
        while True:
            print("=" * 10)
            raw_angles_deg, _ = arm.get_raw_joint_angles()
            if raw_angles_deg is not None:
                for angle_deg in raw_angles_deg:
                    print(f"  - {angle_deg:.2f}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        arm.disconnect_arm()


if __name__ == "__main__":
    main()
