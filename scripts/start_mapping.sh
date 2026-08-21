#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/env.sh"
if [[ ! -f "${project_root}/install/setup.bash" ]]; then
  echo "Build the workspace first with ./scripts/build.sh" >&2
  exit 2
fi
ros2 launch go2_bringup full_stack.launch.py "$@"
