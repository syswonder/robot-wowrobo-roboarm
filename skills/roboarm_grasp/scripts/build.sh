#!/usr/bin/env bash
set -eo pipefail
PKG="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PKG"

mkdir -p rbnx-build/data
rbnx codegen -p "$PKG"
rbnx codegen -p "$PKG" --mcp
echo "[roboarm_grasp] build done"
