#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/env.sh"

if [[ ! -f "${project_root}/install/setup.bash" ]]; then
  echo "Build the workspace first with ./scripts/build.sh" >&2
  exit 2
fi

viewer_requested="true"
for argument in "$@"; do
  if [[ "${argument}" == "use_viewer:=false" ]]; then
    viewer_requested="false"
  fi
done
if [[ "${viewer_requested}" == "true" && -z "${DISPLAY:-}" ]]; then
  echo "The RGB-D viewer needs a graphical DISPLAY." >&2
  echo "Reconnect with ssh -Y unitree@<robot-ip>, or pass use_viewer:=false." >&2
  exit 2
fi

ros2 launch go2_mapping_depthcam depth_mapping.launch.py \
  start_camera:=true \
  start_mapper:=false \
  publish_odom_tf:=false \
  use_viewer:=true \
  "$@"
