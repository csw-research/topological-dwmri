#!/usr/bin/env bash
#SBATCH --job-name=topo_dwmri_hcp_v2
#SBATCH --partition=normal
#SBATCH --time=00:45:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --array=1-30%20
#SBATCH --output=logs/hcp_v2_%A_%a.out
#SBATCH --error=logs/hcp_v2_%A_%a.err

# Per-subject HCP DW-MRI pipeline v2.
# Adds JHU ICBM-DTI-81 atlas ROIs (via FSL flirt) and split-half
# test-retest analogue.

set -euo pipefail

PROJECT="${SCRATCH}/topological-dwmri"
PARENT="${SCRATCH}/non_gaussian_stochastic"
HCP_ROOT="${SCRATCH}/hcp_dwi_data"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate topo_dwmri

# FSL on Sherlock
export FSLDIR=/share/software/user/open/fsl/5.0.10
export PATH=${FSLDIR}/bin:${PATH}
export FSLOUTPUTTYPE=NIFTI_GZ
[ -f "${FSLDIR}/etc/fslconf/fsl.sh" ] && . "${FSLDIR}/etc/fslconf/fsl.sh"

mkdir -p "${PROJECT}/logs" "${PROJECT}/results/hcp_v2"
export TOPO_DWMRI_PARENT_PATH="${PARENT}"

SUBJECT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${PROJECT}/sherlock/subjects.txt" \
    | awk 'NF && !/^#/')
DATA_DIR="${HCP_ROOT}/${SUBJECT}/T1w/Diffusion"

cd "${PROJECT}"
python scripts/hcp_subject_pipeline_v2.py \
    --subject "${SUBJECT}" \
    --data    "${DATA_DIR}/data.nii.gz" \
    --bvals   "${DATA_DIR}/bvals" \
    --bvecs   "${DATA_DIR}/bvecs" \
    --mask    "${DATA_DIR}/nodif_brain_mask.nii.gz" \
    --out     "${PROJECT}/results/hcp_v2/${SUBJECT}" \
    --pooling-radius 1 \
    --downsample 2
