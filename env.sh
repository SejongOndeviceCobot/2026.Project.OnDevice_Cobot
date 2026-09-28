#!/usr/bin/env bash
# Source after the workspace setup/env.sh when it is available.
_cobot_project="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
_cobot_workspace="$(cd "${_cobot_project}/../../.." && pwd -P)"
if [[ -n "${JCLEE_WORKSPACE:-}" && "$(cd "${JCLEE_WORKSPACE}" && pwd -P)" != "${_cobot_workspace}" ]]; then
  echo "JCLEE_WORKSPACE must be the checkout's workspace root" >&2
  return 1
fi
export JCLEE_WORKSPACE="${_cobot_workspace}"
export ISAAC_P0_PROJECT="${_cobot_project}"
export ISAAC_P0_CACHE="${JCLEE_WORKSPACE}/cache/$(basename "${ISAAC_P0_PROJECT}")"
export ISAAC_P0_RUNS="${JCLEE_WORKSPACE}/runs/$(basename "${ISAAC_P0_PROJECT}")"
export ISAAC_P0_DATA="${JCLEE_WORKSPACE}/data/depallet_isaac_p0"
export PYTHONPATH="${ISAAC_P0_PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export UV_CACHE_DIR="${ISAAC_P0_CACHE}/uv"
export UV_PYTHON_INSTALL_DIR="${ISAAC_P0_CACHE}/python"
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
unset _cobot_project _cobot_workspace
