#!/usr/bin/env bash
# Source this file; executing it cannot modify the calling shell.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this script: source scripts/activate_conda.sh" >&2
  exit 2
fi

_go2_conda_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_go2_conda_profile="${_go2_conda_root}/.miniforge3/etc/profile.d/conda.sh"
_go2_conda_prefix="${_go2_conda_root}/.conda/envs/slam_nav"

if [[ ! -f "${_go2_conda_profile}" || ! -x "${_go2_conda_prefix}/bin/python" ]]; then
  echo "Conda environment not found. Run ./scripts/create_conda_env.sh first." >&2
  unset _go2_conda_root _go2_conda_profile _go2_conda_prefix
  return 1
fi

# shellcheck disable=SC1090
source "${_go2_conda_profile}"
conda activate "${_go2_conda_prefix}"
export GO2_SEMANTIC_PYTHON="${CONDA_PREFIX}/bin/python"
# shellcheck disable=SC1091
source "${_go2_conda_root}/scripts/env.sh"
echo "SLAM_nav Conda active: ${CONDA_PREFIX}"

unset _go2_conda_root _go2_conda_profile _go2_conda_prefix
