#!/usr/bin/env bash
set +u
set -eo pipefail
PKG_ROOT="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export RBNX_PACKAGE_ROOT="$PKG_ROOT"
cd "$PKG_ROOT"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
if [[ -f "$ROS_SETUP" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$ROS_SETUP"
fi

ROS2_OVERLAY="$PKG_ROOT/rbnx-build/codegen/ros2_idl/install/setup.bash"
if [[ -f "$ROS2_OVERLAY" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$ROS2_OVERLAY"
else
  echo "[orbbec_camera_roboarm] WARN: ROS2 overlay not found; run rbnx build first" >&2
fi

export ROBONIX_ADVERTISE_HOST="${ROBONIX_ADVERTISE_HOST:-127.0.0.1}"
export PYTHONPATH="$PKG_ROOT/src:$(rbnx path robonix-api):$PKG_ROOT:${PYTHONPATH:-}"

exec python3 -m orbbec_camera_roboarm.main
