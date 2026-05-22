#!/usr/bin/env bash
#SBATCH --job-name=topo_dwmri_retest
#SBATCH --partition=normal
#SBATCH --time=01:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --array=1-44%20
#SBATCH --output=logs/retest_dn_%A_%a.out
#SBATCH --error=logs/retest_dn_%A_%a.err

# Run the v2 pipeline on BOTH visits for each retest subject.
# Subjects come from sherlock/retest_subjects.txt (44 subjects).
# Visit 1: try $SCRATCH/hcp_dwi_data first, fall back to dogun's mirror.
# Visit 2 (retest): always $SCRATCH/hcp_retest.

set -euo pipefail

PROJECT="${SCRATCH}/topological-dwmri"
PARENT="${SCRATCH}/non_gaussian_stochastic"
V1_PRIMARY="${SCRATCH}/hcp_dwi_data"
V1_FALLBACK="/scratch/users/dogun/hcp_dwi_data"
V2_ROOT="${SCRATCH}/hcp_retest"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate topo_dwmri

# FSL
export FSLDIR=/share/software/user/open/fsl/5.0.10
export PATH=${FSLDIR}/bin:${PATH}
export FSLOUTPUTTYPE=NIFTI_GZ
[ -f "${FSLDIR}/etc/fslconf/fsl.sh" ] && . "${FSLDIR}/etc/fslconf/fsl.sh"

mkdir -p "${PROJECT}/logs" "${PROJECT}/results/retest_v1_dn" "${PROJECT}/results/retest_v2_dn"
export TOPO_DWMRI_PARENT_PATH="${PARENT}"

SUBJECT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${PROJECT}/sherlock/retest_subjects.txt" \
    | awk 'NF && !/^#/')

# Resolve visit-1 path
V1_DIR="${V1_PRIMARY}/${SUBJECT}/T1w/Diffusion"
if [[ ! -f "${V1_DIR}/data.nii.gz" ]]; then
    V1_DIR="${V1_FALLBACK}/${SUBJECT}/T1w/Diffusion"
fi
V2_DIR="${V2_ROOT}/${SUBJECT}/T1w/Diffusion"

echo "=== ${SUBJECT} ==="
echo "  V1: ${V1_DIR}"
echo "  V2: ${V2_DIR}"

cd "${PROJECT}"

# Visit 1
python scripts/hcp_subject_pipeline_v2.py \
    --subject "${SUBJECT}" \
    --data    "${V1_DIR}/data.nii.gz" \
    --bvals   "${V1_DIR}/bvals" \
    --bvecs   "${V1_DIR}/bvecs" \
    --mask    "${V1_DIR}/nodif_brain_mask.nii.gz" \
    --out     "${PROJECT}/results/retest_v1_dn/${SUBJECT}" \
    --pooling-radius 1 \
    --downsample 2 \
    --skip-splits \
    --denoise

# Visit 2 (retest)
python scripts/hcp_subject_pipeline_v2.py \
    --subject "${SUBJECT}" \
    --data    "${V2_DIR}/data.nii.gz" \
    --bvals   "${V2_DIR}/bvals" \
    --bvecs   "${V2_DIR}/bvecs" \
    --mask    "${V2_DIR}/nodif_brain_mask.nii.gz" \
    --out     "${PROJECT}/results/retest_v2_dn/${SUBJECT}" \
    --pooling-radius 1 \
    --downsample 2 \
    --skip-splits \
    --denoise
