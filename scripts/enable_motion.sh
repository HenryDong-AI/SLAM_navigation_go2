#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--i-understand" ]]; then
  echo "Motion stays disarmed by design." >&2
  echo "Clear the area, hold the physical controller/E-stop, verify RViz and LiDAR," >&2
  echo "then run: $0 --i-understand" >&2
  exit 2
fi
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/env.sh"
ros2 service call /go2/motion/enable std_srvs/srv/SetBool "{data: true}"
