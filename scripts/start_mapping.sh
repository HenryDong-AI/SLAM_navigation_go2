#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/env.sh"
if [[ ! -f "${project_root}/install/setup.bash" ]]; then
  echo "Build the workspace first with ./scripts/build.sh" >&2
  exit 2
fi

# This process check also catches a stack which was started before the lock was
# introduced and therefore cannot own the lock below.
existing_launch_pids="$(
  pgrep -f '/opt/ros/foxy/bin/ros2 launch go2_bringup full_stack.launch.py' \
    || true
)"
if [[ -n "${existing_launch_pids}" ]]; then
  echo "Another complete stack is already running (launch PID(s):" >&2
  echo "${existing_launch_pids})." >&2
  echo "Stop it completely before starting a replacement." >&2
  exit 3
fi

# A second sensor_time_bridge has a different process identity. Every safety
# consumer deliberately latches that change, so never allow two complete
# stacks to overlap.
stack_lock="/tmp/slam_nav_${UID}_full_stack.lock"
exec 9>"${stack_lock}"
if ! flock -n 9; then
  echo "Another SLAM_nav complete stack already holds ${stack_lock}." >&2
  echo "Stop its terminal with Ctrl-C before starting a replacement." >&2
  exit 3
fi

time_sync_info="$(ros2 topic info /go2/time_sync/status 2>/dev/null || true)"
if printf '%s\n' "${time_sync_info}" | rg -q '^Publisher count: [1-9][0-9]*$'; then
  echo "A /go2/time_sync/status publisher is already running." >&2
  echo "Starting another complete stack would latch a sensor-time fault." >&2
  echo "Stop the existing complete stack, wait for its publisher to disappear," >&2
  echo "then run this command again." >&2
  exit 3
fi

ros2 launch go2_bringup full_stack.launch.py "$@"
