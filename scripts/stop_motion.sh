#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/env.sh"
ros2 service call /go2/motion/stop std_srvs/srv/Trigger "{}"
