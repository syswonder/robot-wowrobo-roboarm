#!/usr/bin/env bash
set -eo pipefail
PKG="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

rbnx codegen -p "$PKG"
rbnx codegen -p "$PKG" --ros2

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
if [[ -f "$ROS_SETUP" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$ROS_SETUP"
  ( cd "$PKG/rbnx-build/codegen/ros2_idl" && colcon build )
else
  echo "[orbbec_camera_roboarm] WARN: ROS setup not found at $ROS_SETUP; skip colcon build" >&2
fi

echo "[orbbec_camera_roboarm] build done"
