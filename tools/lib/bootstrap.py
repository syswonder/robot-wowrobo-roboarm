"""Bootstrap PYTHONPATH for standalone SDK tools (no Robonix / ROS)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from lib.paths import (
    camera_src_root,
    deploy_root,
    lerobot_src_root,
    skill_assets_root,
    tools_root,
)


def setup() -> Path:
    root = deploy_root()
    for entry in (
        tools_root(),
        lerobot_src_root(),
        camera_src_root(),
    ):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)

    from lib.config import init_config

    config_path = Path(
        os.environ.get("TOOLS_CONFIG", tools_root() / "config.yaml")
    ).resolve()
    init_config(str(config_path), assets_root=skill_assets_root())
    return root
