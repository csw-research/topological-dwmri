#!/usr/bin/env bash
#SBATCH --job-name=topo_dwmri_hcp
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=2
#SBATCH --array=1-30%20
#SBATCH --output=logs/hcp_%A_%a.out
#SBATCH --error=logs/hcp_%A_%a.err

# Per-subject HCP DW-MRI pipeline (FA / MD / MK / alpha_persistence).
# Reads subject IDs from sherlock/subjects.txt.

set -euo pipefail

PROJECT="${SCRATCH}/topological-dwmri"
PARENT="${SCRATCH}/non_gaussian_stochastic"
HCP_ROOT="${SCRATCH}/hcp_dwi_data"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate topo_dwmri

mkdir -p "${PROJECT}/logs" "${PROJECT}/results/hcp"
export TOPO_DWMRI_PARENT_PATH="${PARENT}"

SUBJECT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${PROJECT}/sherlock/subjects.txt" \
    | awk 'NF && !/^#/')
DATA_DIR="${HCP_ROOT}/${SUBJECT}/T1w/Diffusion"

cd "${PROJECT}"
python scripts/hcp_subject_pipeline.py \
    --subject "${SUBJECT}" \
    --data    "${DATA_DIR}/data.nii.gz" \
    --bvals   "${DATA_DIR}/bvals" \
    --bvecs   "${DATA_DIR}/bvecs" \
    --mask    "${DATA_DIR}/nodif_brain_mask.nii.gz" \
    --out     "${PROJECT}/results/hcp/${SUBJECT}" \
    --pooling-radius 1 \
    --downsample 2
