#!/usr/bin/env python3
"""roboarm_sort — 按 C/O/S 字母顺序编排抓取与放置（直接调用 roboarm_core）。"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from robonix_api import Err, Ok, Skill
from sort_core.config import init_config
from sort_core import grasp_client
from sort_core.pipeline import sort_letters

log = logging.getLogger("roboarm_sort")

skill = Skill(id="roboarm_sort", namespace="robonix/skill/roboarm_sort")

_cfg: dict[str, Any] = {}
_pkg_root: Path | None = None


def _resolve_pkg_root(cfg: dict) -> Path:
    raw = cfg.get("package_root") or os.environ.get("RBNX_PACKAGE_ROOT")
    if raw:
        path = Path(str(raw)).resolve()
        if not path.is_dir():
            raise ValueError(f"package_root does not exist: {path}")
        return path
    return Path(__file__).resolve().parent.parent


@skill.on_init
def init(cfg: dict):
    global _cfg, _pkg_root
    _cfg = dict(cfg)
    try:
        _pkg_root = _resolve_pkg_root(_cfg)
    except ValueError as exc:
        return Err(str(exc))

    config_yaml = _cfg.get("config_yaml") or str(_pkg_root / "config" / "config.yaml")
    try:
        init_config(config_yaml)
        grasp_client.load_grasp_config(_cfg, _pkg_root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        return Err(f"Failed to load config: {exc}")

    log.info("roboarm_sort initialized (sort_config=%s)", config_yaml)
    return Ok()


@skill.on_activate
def activate():
    assert _pkg_root is not None
    try:
        grasp_client.connect_grasp_runtime(skill, _cfg, _pkg_root)
    except Exception as exc:
        return Err(f"Failed to connect grasp runtime: {exc}")
    log.info("roboarm_sort activated")
    return Ok()


@skill.on_deactivate
def deactivate():
    grasp_client.close_grasp_client()
    log.info("roboarm_sort deactivated")
    return Ok()


try:
    from roboarm_sort_mcp import SortLetters_Request, SortLetters_Response  # type: ignore
    from std_msgs_mcp import String
except ImportError:  # pragma: no cover
    SortLetters_Request = SortLetters_Response = None  # type: ignore
    String = None  # type: ignore


if SortLetters_Request is not None:

    @skill.mcp("robonix/skill/roboarm_sort/sort_letters")
    def sort_letters_tool(req: SortLetters_Request) -> SortLetters_Response:
        order = (req.order.data or "").strip()
        if not order:
            return SortLetters_Response(
                success=False,
                message=String(data="order 不能为空"),
            )
        try:
            result = sort_letters(order)
        except Exception as exc:
            log.error("sort_letters 异常: %s", exc, exc_info=True)
            return SortLetters_Response(
                success=False,
                message=String(data=str(exc)),
            )

        success = result.get("status") == "success"
        details = "; ".join(result.get("details") or [])
        msg = f"order={order}; {details}" if details else f"order={order}"
        if result.get("reason"):
            msg += f"; {result['reason']}"
        return SortLetters_Response(
            success=success,
            message=String(data=msg),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    skill.run()
