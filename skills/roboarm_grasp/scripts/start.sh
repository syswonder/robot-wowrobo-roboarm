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

SKILL_OVERLAY="$PKG_ROOT/rbnx-build/codegen/ros2_idl/install/setup.bash"
ARM_OVERLAY="$PKG_ROOT/../../primitives/roboarm_arm/rbnx-build/codegen/ros2_idl/install/setup.bash"
if [[ -f "$SKILL_OVERLAY" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$SKILL_OVERLAY"
elif [[ -f "$ARM_OVERLAY" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$ARM_OVERLAY"
else
  echo "[roboarm_grasp] WARN: ROS2 overlay not found; run rbnx build first" >&2
fi

export ROBONIX_ADVERTISE_HOST="${ROBONIX_ADVERTISE_HOST:-127.0.0.1}"
export PYTHONPATH="$PKG_ROOT/lib:$(rbnx path robonix-api):$PKG_ROOT:${PYTHONPATH:-}"

exec python3 -m roboarm_grasp.main
