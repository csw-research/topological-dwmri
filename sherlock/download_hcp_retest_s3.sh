#!/usr/bin/env bash
# Download the diffusion subset of HCP-YA Retest 2025 from the HCP open-access
# S3 bucket into $SCRATCH/hcp_retest/.
#
# Requires:
#   - AWS CLI installed (aws --version)
#   - HCP credentials at $HOME/.aws_hcp_credentials in INI format with profile [hcp]
#
# Per-subject footprint: ~1.2 GB (data.nii.gz) + 100 KB metadata. 45 subjects
# total ~= 55 GB. Network: well under an hour on Sherlock's S3 link.

set -euo pipefail

PROJECT_DEST="${SCRATCH}/hcp_retest"
mkdir -p "${PROJECT_DEST}"

export AWS_SHARED_CREDENTIALS_FILE="${HOME}/.aws_hcp_credentials"
export AWS_PROFILE=hcp

# Source bucket
BUCKET="s3://hcp-openaccess/HCP_Retest"

# Discover subject list from the bucket itself (no need to maintain a separate
# subjects file).
echo "Discovering retest subjects..."
SUBJECTS=$(aws s3 ls "${BUCKET}/" | awk '/PRE / {sub("/","",$2); print $2}')
N=$(echo "${SUBJECTS}" | wc -l)
echo "Found ${N} subjects."

# We only need the T1w/Diffusion subset. Pull exactly the four files plus
# eddylogs/ which contains the q-space gradient field corrections (optional).
COUNTER=0
for s in ${SUBJECTS}; do
    COUNTER=$((COUNTER + 1))
    SRC="${BUCKET}/${s}/T1w/Diffusion/"
    DEST="${PROJECT_DEST}/${s}/T1w/Diffusion/"
    if [[ -f "${DEST}/data.nii.gz" && -f "${DEST}/bvals" ]]; then
        echo "[${COUNTER}/${N}] ${s}: already present, skip"
        continue
    fi
    echo "[${COUNTER}/${N}] ${s}: syncing..."
    aws s3 sync "${SRC}" "${DEST}" \
        --exclude "eddylogs/*" \
        --only-show-errors
done

echo "Done. Total subjects in ${PROJECT_DEST}:"
ls -1 "${PROJECT_DEST}" | wc -l
du -sh "${PROJECT_DEST}"
