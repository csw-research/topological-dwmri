#!/usr/bin/env bash
#SBATCH --job-name=topo_dwmri_sim
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/sim_%A.out
#SBATCH --error=logs/sim_%A.err

# Run the full-scale simulation study (experiments 1--4 at production sizes).
# Submit with: sbatch sherlock/run_simulation.sbatch

set -euo pipefail
PROJECT="${SCRATCH}/topological-dwmri"
PARENT="${SCRATCH}/non_gaussian_stochastic"

module purge
module load python/3.11.2 py-pip/22.2.2_py311
# shellcheck disable=SC1091
source "${SCRATCH}/envs/topological_dwmri/bin/activate"

mkdir -p "${PROJECT}/logs" "${PROJECT}/results"
export TOPO_DWMRI_PARENT_PATH="${PARENT}"

cd "${PROJECT}"
python scripts/run_all_experiments.py --out-dir "${PROJECT}/results"
python analysis/make_figures.py
