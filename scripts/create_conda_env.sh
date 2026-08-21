#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
miniforge_version="26.3.2-3"
miniforge_installer="Miniforge3-${miniforge_version}-Linux-aarch64.sh"
miniforge_url="https://github.com/conda-forge/miniforge/releases/download/${miniforge_version}/${miniforge_installer}"
miniforge_sha256="2c113a69297e612b01ca0f320c22a3107a11f2ab9b573d79ac868a175945ce29"
miniforge_prefix="${project_root}/.miniforge3"
env_prefix="${project_root}/.conda/envs/slam_nav"
source_site="/home/unitree/Documents/demov1/venv-yolo/lib/python3.8/site-packages"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This environment is pinned for the Go2 Jetson (aarch64)." >&2
  exit 1
fi

if [[ ! -d "${source_site}" ]]; then
  echo "The validated Jetson YOLO environment is missing: ${source_site}" >&2
  exit 1
fi

if [[ ! -x "${miniforge_prefix}/bin/conda" ]]; then
  installer_path="$(mktemp "/tmp/${miniforge_installer}.XXXXXX")"
  trap 'rm -f "${installer_path:-}"' EXIT
  echo "Downloading Miniforge ${miniforge_version} for ARM64..."
  curl --fail --location --retry 3 --output "${installer_path}" "${miniforge_url}"
  echo "${miniforge_sha256}  ${installer_path}" | sha256sum --check --status
  bash "${installer_path}" -b -p "${miniforge_prefix}"
fi

if [[ ! -x "${env_prefix}/bin/python" ]]; then
  "${miniforge_prefix}/bin/conda" env create \
    --prefix "${env_prefix}" \
    --file "${project_root}/environment.conda.yml" \
    --yes
else
  "${miniforge_prefix}/bin/conda" env update \
    --prefix "${env_prefix}" \
    --file "${project_root}/environment.conda.yml" \
    --prune
fi

"${miniforge_prefix}/bin/conda" config --add envs_dirs "${project_root}/.conda/envs"
"${miniforge_prefix}/bin/conda" config --set auto_activate false
"${miniforge_prefix}/bin/conda" init bash

activate_dir="${env_prefix}/etc/conda/activate.d"
mkdir -p "${activate_dir}"
install -m 0644 \
  "${project_root}/scripts/conda_activate_hook.sh" \
  "${activate_dir}/go2_slam_nav.sh"

target_site="$("${env_prefix}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
echo "Mirroring the device-validated Jetson Torch/Ultralytics packages..."
rsync -a \
  --exclude 'pip/' \
  --exclude 'pip-*' \
  --exclude 'setuptools/' \
  --exclude 'setuptools-*' \
  --exclude 'pkg_resources/' \
  --exclude 'wheel/' \
  --exclude 'wheel-*' \
  --exclude '_virtualenv.py' \
  --exclude '_virtualenv.pth' \
  --exclude 'distutils-precedence.pth' \
  "${source_site}/" "${target_site}/"

# Foxy, cv_bridge, and OpenCV are device packages compiled for Python 3.8.
printf '%s\n' \
  '/usr/local/lib/python3.8/dist-packages' \
  '/usr/lib/python3/dist-packages' \
  '/usr/lib/python3.8/dist-packages' \
  '/usr/lib/python3.8/site-packages' \
  > "${target_site}/go2_system_site_packages.pth"

export GO2_SEMANTIC_PYTHON="${env_prefix}/bin/python"
# shellcheck disable=SC1091
source "${project_root}/scripts/env.sh"
"${env_prefix}/bin/python" -c \
  'import sys, numpy, cv2, cv_bridge, rclpy, torch, ultralytics; assert torch.cuda.is_available(); print("Validated Python {}, Torch {}, Ultralytics {}, CUDA available".format(sys.version.split()[0], torch.__version__, ultralytics.__version__))'

echo
echo "Conda environment ready: ${env_prefix}"
echo "Open a new terminal, then activate it with: conda activate slam_nav"
