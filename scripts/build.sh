#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/env.sh"
cd "${project_root}"
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
