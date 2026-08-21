#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/map.yaml [launch arguments...]" >&2
  exit 2
fi
map_path="$1"
shift
if [[ ! -f "${map_path}" ]]; then
  echo "Map YAML not found: ${map_path}" >&2
  exit 2
fi
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/env.sh"
ros2 launch go2_bringup localization_stack.launch.py map:="${map_path}" "$@"
