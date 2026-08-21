#!/usr/bin/env bash
# Installed into the named Conda environment by create_conda_env.sh.

_go2_conda_project_root="$(cd "${CONDA_PREFIX}/../../.." && pwd)"
export GO2_SEMANTIC_PYTHON="${CONDA_PREFIX}/bin/python"
# shellcheck disable=SC1091
source "${_go2_conda_project_root}/scripts/env.sh"
unset _go2_conda_project_root
