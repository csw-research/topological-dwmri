#!/usr/bin/env bash
# Environment setup for the topological-dwmri project on Sherlock.
# Run once per cluster login.
set -euo pipefail

PROJECT_NAME="topological_dwmri"
ENV_DIR="${SCRATCH}/envs/${PROJECT_NAME}"

module purge
module load python/3.11.2
module load py-pip/22.2.2_py311

if [[ ! -d "${ENV_DIR}" ]]; then
    python -m venv "${ENV_DIR}"
fi

# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"
pip install --upgrade pip
pip install -r "$(dirname "${BASH_SOURCE[0]}")/requirements.txt"

echo "Activated venv at ${ENV_DIR}"
echo "To re-activate in another shell:"
echo "  module load python/3.11.2 py-pip/22.2.2_py311"
echo "  source ${ENV_DIR}/bin/activate"
