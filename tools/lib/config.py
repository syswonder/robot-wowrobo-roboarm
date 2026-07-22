"""Tool-local configuration loader (no Robonix / ROS dependencies)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG: dict[str, Any] = {}
_CONFIG_PATH: Path | None = None
_ASSETS_ROOT: Path | None = None


def load_config(config_file: str) -> dict[str, Any]:
    path = Path(config_file)
    if not path.is_file():
        raise FileNotFoundError(f"配置文件未找到: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def init_config(
    config_file: str,
    *,
    assets_root: str | Path | None = None,
) -> dict[str, Any]:
    global CONFIG, _CONFIG_PATH, _ASSETS_ROOT
    _CONFIG_PATH = Path(config_file).resolve()
    CONFIG = load_config(str(_CONFIG_PATH))

    grasp_config = CONFIG.get("grasp_config")
    if grasp_config:
        grasp_path = Path(str(grasp_config))
        if not grasp_path.is_absolute():
            grasp_path = (_CONFIG_PATH.parent / grasp_path).resolve()
        if grasp_path.is_file():
            overlay = load_config(str(grasp_path))
            CONFIG = {**overlay, **CONFIG}

    _ASSETS_ROOT = (
        Path(assets_root).resolve()
        if assets_root
        else _CONFIG_PATH.parent.parent / "skills/roboarm_grasp/assets"
    )
    return CONFIG


def get_config_dir() -> Path:
    if _CONFIG_PATH is None:
        raise RuntimeError("Config not initialized; call init_config() first")
    return _CONFIG_PATH.parent


def get_assets_root() -> Path:
    if _ASSETS_ROOT is None:
        raise RuntimeError("Assets root not initialized; call init_config() first")
    return _ASSETS_ROOT


def resolve_asset(relative: str) -> Path:
    return get_assets_root() / relative


def get_config_value(
    key: str,
    default: Any = None,
    *,
    raise_if_missing: bool = True,
) -> Any:
    if raise_if_missing and key not in CONFIG:
        raise KeyError(f"配置项未找到: {key}")
    return CONFIG.get(key, default)
