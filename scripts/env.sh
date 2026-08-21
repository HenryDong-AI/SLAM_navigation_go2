#!/usr/bin/env bash
# Source this file: `source scripts/env.sh`

_go2_env_restore_nounset=0
case "$-" in
  *u*)
    _go2_env_restore_nounset=1
    # Generated Foxy setup files read optional variables without ${var:-}.
    set +u
    ;;
esac

_go2_env_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f /opt/ros/foxy/setup.bash ]]; then
  echo "ROS 2 Foxy was not found at /opt/ros/foxy" >&2
  if [[ "${_go2_env_restore_nounset}" == "1" ]]; then
    set -u
  fi
  return 1 2>/dev/null || exit 1
fi

source /opt/ros/foxy/setup.bash

if [[ -f /unitree/module/graph_pid_ws/install/setup.bash ]]; then
  source /unitree/module/graph_pid_ws/install/setup.bash
fi

if [[ -f /home/unitree/cyclonedds_ws/install/setup.bash ]]; then
  source /home/unitree/cyclonedds_ws/install/setup.bash
fi

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file:///home/unitree/cyclonedds_ws/cyclonedds.xml}"
export GO2_NETWORK_INTERFACE="${GO2_NETWORK_INTERFACE:-eth0}"
export GO2_SDK_PYTHON="${GO2_SDK_PYTHON:-/home/unitree/Documents/demov1/unitree_sdk2_python}"
_go2_conda_semantic="${_go2_env_root}/.conda/envs/slam_nav/bin/python"
if [[ -z "${GO2_SEMANTIC_PYTHON:-}" ]]; then
  if [[ -x "${_go2_conda_semantic}" ]]; then
    GO2_SEMANTIC_PYTHON="${_go2_conda_semantic}"
  else
    GO2_SEMANTIC_PYTHON="/home/unitree/Documents/demov1/venv-yolo/bin/python"
  fi
  export GO2_SEMANTIC_PYTHON
fi
if [[ -z "${GO2_CYCLONEDDS_PYTHON:-}" ]]; then
  GO2_CYCLONEDDS_PYTHON="$(python3 -m site --user-site)"
  export GO2_CYCLONEDDS_PYTHON
fi

_go2_noshm_lib="${GO2_SDK_PYTHON}/cyclonedds/install_noshm/lib"
if [[ -d "${_go2_noshm_lib}" ]]; then
  export LD_LIBRARY_PATH="${_go2_noshm_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export PYTHONPATH="${GO2_SDK_PYTHON}:${GO2_CYCLONEDDS_PYTHON}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -f "${_go2_env_root}/install/setup.bash" ]]; then
  source "${_go2_env_root}/install/setup.bash"
fi

unset _go2_noshm_lib
unset _go2_conda_semantic
unset _go2_env_root
if [[ "${_go2_env_restore_nounset}" == "1" ]]; then
  unset _go2_env_restore_nounset
  set -u
else
  unset _go2_env_restore_nounset
fi
