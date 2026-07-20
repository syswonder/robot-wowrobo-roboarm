#!/usr/bin/env bash
# Offline build: codegen gRPC/Python stubs and Robonix ROS 2 IDL overlay.
set -eo pipefail
PKG="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

# gRPC stubs (driver lifecycle + any @provider.grpc handlers)
rbnx codegen -p "$PKG"
# ROS 2 IDL overlay for topic payloads
rbnx codegen -p "$PKG" --ros2

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
if [[ -f "$ROS_SETUP" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$ROS_SETUP"
  ( cd "$PKG/rbnx-build/codegen/ros2_idl" && colcon build )
else
  echo "[roboarm_arm] WARN: ROS setup not found at $ROS_SETUP; skip colcon build" >&2
fi

echo "[roboarm_arm] build done"
