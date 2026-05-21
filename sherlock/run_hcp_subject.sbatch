#!/usr/bin/env bash
#SBATCH --job-name=topo_dwmri_hcp
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-20
#SBATCH --output=logs/hcp_%A_%a.out
#SBATCH --error=logs/hcp_%A_%a.err

# Per-subject voxelwise persistence-tail map for the HCP dataset.
# Expects a subject list at sherlock/subjects.txt (one HCP ID per line).
# Submit with: sbatch sherlock/run_hcp_subject.sbatch

set -euo pipefail

PROJECT="${SCRATCH}/topological-dwmri"
PARENT="${SCRATCH}/non_gaussian_stochastic"
HCP_ROOT="${OAK}/users/${USER}/HCP_Diffusion"   # adjust to local layout

module purge
module load python/3.11.2 py-pip/22.2.2_py311
# shellcheck disable=SC1091
source "${SCRATCH}/envs/topological_dwmri/bin/activate"

mkdir -p "${PROJECT}/logs" "${PROJECT}/results/hcp"
export TOPO_DWMRI_PARENT_PATH="${PARENT}"

SUBJECT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${PROJECT}/sherlock/subjects.txt")
DATA_DIR="${HCP_ROOT}/${SUBJECT}/T1w/Diffusion"

cd "${PROJECT}"
python scripts/process_subject.py \
    --subject "${SUBJECT}" \
    --data    "${DATA_DIR}/data.nii.gz" \
    --bvals   "${DATA_DIR}/bvals" \
    --bvecs   "${DATA_DIR}/bvecs" \
    --mask    "${DATA_DIR}/nodif_brain_mask.nii.gz" \
    --out     "${PROJECT}/results/hcp/${SUBJECT}"
