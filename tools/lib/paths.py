"""Path helpers for robonix/roboarm standalone tools."""

from __future__ import annotations

from pathlib import Path


def tools_root() -> Path:
    return Path(__file__).resolve().parent.parent


def deploy_root() -> Path:
    return tools_root().parent


def arm_assets_root() -> Path:
    return deploy_root() / "primitives/roboarm_arm/src/assets"


def skill_assets_root() -> Path:
    return deploy_root() / "skills/roboarm_grasp/assets"


def lerobot_src_root() -> Path:
    return deploy_root() / "primitives/roboarm_arm/src/vendor/lerobot/src"


def camera_src_root() -> Path:
    return deploy_root() / "primitives/orbbec_camera_roboarm/src"


def urdf_path() -> Path:
    return arm_assets_root() / "urdf/lerobo/low_cost_robot.urdf"


def calibration_dir() -> Path:
    return arm_assets_root() / "calibration"


def hand_eye_dir() -> Path:
    return skill_assets_root() / "hand_eye"


def hand_eye_matrix_path() -> Path:
    return hand_eye_dir() / "2d_homography.npy"
