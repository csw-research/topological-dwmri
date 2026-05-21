#!/usr/bin/env python3
"""HCP single-subject pipeline.

For one HCP subject:

  1. Load data.nii.gz / bvals / bvecs / nodif_brain_mask.nii.gz.
  2. Fit DTI (FA, MD) via dipy.
  3. Fit DKI (MK) via dipy.
  4. Derive WM / GM / CSF tissue labels from FA and mean b=0.
  5. Compute voxelwise alpha_persistence using the cumulative-bridge
     construction and the parent paper's persistence library.
  6. Compute per-ROI summary statistics (corpus callosum approximated by
     a high-FA midline mask; cortical GM, deep GM, CSF, WM).
  7. Save four NIfTI maps and a per-ROI JSON summary.

Outputs (under --out):

    <subject>_FA.nii.gz
    <subject>_MD.nii.gz
    <subject>_MK.nii.gz
    <subject>_alpha_persistence.nii.gz
    <subject>_tissue.nii.gz   (1=WM, 2=GM, 3=CSF, 4=corpus_callosum)
    <subject>_roi_summary.json

Designed to be submitted as a SLURM array job.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.parent_bridge import (
    sublevel_persistence_1d,
    persistence_lifetimes,
)


# ---------------------------------------------------------------------------
# Utility wrappers around dipy
# ---------------------------------------------------------------------------

def fit_dti(data, mask, bvals, bvecs):
    """Return FA and MD maps using dipy's tensor model."""
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)
    tm = TensorModel(gtab)
    fit = tm.fit(data, mask=mask)
    return fit.fa.astype(np.float32), fit.md.astype(np.float32)


def fit_dki(data, mask, bvals, bvecs):
    """Return MK (mean kurtosis) using dipy's diffusion-kurtosis model."""
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dki import DiffusionKurtosisModel
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)
    km = DiffusionKurtosisModel(gtab)
    fit = km.fit(data, mask=mask)
    mk = fit.mk(min_kurtosis=-0.5, max_kurtosis=3.0)
    return mk.astype(np.float32)


def segment_tissue(FA, mean_b0, mask):
    """Coarse WM / GM / CSF segmentation.

    Heuristic thresholds standard in the QC literature; not a substitute
    for FreeSurfer but sufficient for ROI-level comparisons in HCP data.
    """
    tissue = np.zeros_like(FA, dtype=np.uint8)
    inside = mask
    # CSF: hyperintense at b=0 and very low FA
    b0_norm = mean_b0 / (np.percentile(mean_b0[inside], 99) + 1e-6)
    csf = inside & (b0_norm > 0.8) & (FA < 0.15)
    # WM: high FA
    wm = inside & (FA >= 0.4) & ~csf
    # GM: everything else inside the mask
    gm = inside & ~csf & ~wm
    tissue[wm] = 1
    tissue[gm] = 2
    tissue[csf] = 3
    return tissue


def corpus_callosum_mask(tissue, FA):
    """Approximate corpus callosum: high-FA WM along the midsagittal plane."""
    cc = np.zeros_like(tissue, dtype=bool)
    midx = tissue.shape[0] // 2
    band = slice(midx - 2, midx + 3)
    cc[band] = (tissue[band] == 1) & (FA[band] >= 0.6)
    return cc


# ---------------------------------------------------------------------------
# Voxelwise persistence-tail map
# ---------------------------------------------------------------------------

def voxelwise_alpha_persistence(
    data: np.ndarray,
    mask: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    shells=(1000.0, 2000.0, 3000.0),
    tol: float = 50.0,
    pooling_neighbours: int = 0,
    k_fraction: float = 0.15,
    min_k: int = 8,
) -> np.ndarray:
    """Voxelwise persistence-tail exponent map.

    For each voxel and each shell we form the per-direction attenuation
    y_i = -ln(S_i / S0) (with S0 the voxel-mean b=0), centre it, and take
    the cumulative sum Z_j = sum_{i <= j} (y_i - mean_i y_i). Sublevel
    persistence of Z yields lifetimes whose tail exponent is alpha. We
    pool lifetimes across the three shells for each voxel.

    With ``pooling_neighbours > 0`` the lifetimes from the voxel and its
    6-connected spatial neighbours are pooled before applying the Hill
    estimator. This stabilises the per-voxel estimate at the cost of
    spatial resolution.
    """
    X, Y, Z, _ = data.shape
    # mean b=0
    b0_idx = np.where(bvals <= tol)[0]
    if b0_idx.size == 0:
        raise RuntimeError("No b=0 volumes found")
    b0_mean = data[..., b0_idx].mean(axis=-1).astype(np.float32)
    b0_safe = np.maximum(b0_mean, 1.0)
    # For each shell, the indices of the gradient directions
    shell_idx = {}
    for s in shells:
        idx = np.where(np.abs(bvals - s) <= tol)[0]
        if idx.size == 0:
            continue
        shell_idx[s] = idx

    inds = np.argwhere(mask)
    out = np.full(mask.shape, np.nan, dtype=np.float32)

    # Build per-voxel attenuation arrays once
    per_voxel_lifetimes = [[] for _ in range(inds.shape[0])]
    for s, idx in shell_idx.items():
        shell_vols = data[..., idx]  # (X, Y, Z, n_dirs)
        attn = -np.log(np.maximum(shell_vols / b0_safe[..., None], 1e-6))
        for k, (xi, yi, zi) in enumerate(inds):
            y = attn[xi, yi, zi]
            y = y - y.mean()
            if not np.all(np.isfinite(y)):
                continue
            Zk = np.cumsum(y)
            diag = sublevel_persistence_1d(Zk)
            life = persistence_lifetimes(diag)
            if life.size > 0:
                per_voxel_lifetimes[k].append(life)

    # Optionally pool neighbours
    if pooling_neighbours > 0:
        # build voxel-key dict
        loc = {tuple(inds[k]): k for k in range(inds.shape[0])}
        rad = pooling_neighbours
        for k, (xi, yi, zi) in enumerate(inds):
            collected = list(per_voxel_lifetimes[k])
            for dx in range(-rad, rad + 1):
                for dy in range(-rad, rad + 1):
                    for dz in range(-rad, rad + 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        j = loc.get((xi + dx, yi + dy, zi + dz))
                        if j is not None:
                            collected.extend(per_voxel_lifetimes[j])
            if not collected:
                continue
            L = np.sort(np.concatenate(collected))[::-1]
            if L.size < min_k + 1:
                continue
            N = L.size
            k_use = max(min_k, int(k_fraction * N))
            k_use = min(k_use, N - 1)
            m = float(np.mean(np.log(L[:k_use]) - np.log(L[k_use])))
            if m > 0:
                out[xi, yi, zi] = 1.0 / m
    else:
        for k, (xi, yi, zi) in enumerate(inds):
            if not per_voxel_lifetimes[k]:
                continue
            L = np.sort(np.concatenate(per_voxel_lifetimes[k]))[::-1]
            if L.size < min_k + 1:
                continue
            N = L.size
            k_use = max(min_k, int(k_fraction * N))
            k_use = min(k_use, N - 1)
            m = float(np.mean(np.log(L[:k_use]) - np.log(L[k_use])))
            if m > 0:
                out[xi, yi, zi] = 1.0 / m
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True)
    p.add_argument("--data",    required=True)
    p.add_argument("--bvals",   required=True)
    p.add_argument("--bvecs",   required=True)
    p.add_argument("--mask",    required=True)
    p.add_argument("--out",     required=True)
    p.add_argument("--pooling-radius", type=int, default=1,
                   help="3D spatial pooling radius for persistence Hill.")
    p.add_argument("--downsample", type=int, default=2,
                   help="Slice subsampling along each axis to keep the "
                        "persistence step tractable. 1 = full resolution.")
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    print(f"[{args.subject}] loading...", flush=True)
    img = nib.load(args.data)
    data = np.asarray(img.dataobj, dtype=np.float32)
    affine = img.affine
    bvals = np.loadtxt(args.bvals)
    bvecs = np.loadtxt(args.bvecs)
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T
    mask = np.asarray(nib.load(args.mask).dataobj).astype(bool)
    print(f"  data : {data.shape}, mask voxels: {int(mask.sum())}", flush=True)

    # optional downsample
    if args.downsample > 1:
        s = args.downsample
        data = data[::s, ::s, ::s]
        mask = mask[::s, ::s, ::s]
        # adjust affine for downsampling
        scale = np.eye(4)
        scale[:3, :3] *= s
        affine = affine @ scale
        print(f"  downsampled to {data.shape}", flush=True)

    print(f"[{args.subject}] DTI fit (FA, MD)...", flush=True)
    FA, MD = fit_dti(data, mask, bvals, bvecs)

    print(f"[{args.subject}] DKI fit (MK)...", flush=True)
    try:
        MK = fit_dki(data, mask, bvals, bvecs)
    except Exception as e:
        print(f"  DKI failed ({e!r}); writing zeros", flush=True)
        MK = np.zeros_like(FA)

    print(f"[{args.subject}] tissue segmentation...", flush=True)
    b0_idx = np.where(bvals <= 50)[0]
    b0_mean = data[..., b0_idx].mean(axis=-1)
    tissue = segment_tissue(FA, b0_mean, mask)
    cc = corpus_callosum_mask(tissue, FA)
    tissue[cc] = 4
    print(f"  WM voxels : {int((tissue == 1).sum())}", flush=True)
    print(f"  GM voxels : {int((tissue == 2).sum())}", flush=True)
    print(f"  CSF voxels: {int((tissue == 3).sum())}", flush=True)
    print(f"  CC voxels : {int((tissue == 4).sum())}", flush=True)

    print(f"[{args.subject}] voxelwise alpha_persistence...", flush=True)
    alpha = voxelwise_alpha_persistence(
        data, mask, bvals, bvecs,
        pooling_neighbours=args.pooling_radius,
    )

    # Save NIfTIs
    def save_map(arr, name):
        out = out_dir / f"{args.subject}_{name}.nii.gz"
        nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), str(out))
        return out

    for arr, name in [(FA, "FA"), (MD, "MD"), (MK, "MK"),
                       (alpha, "alpha_persistence")]:
        path = save_map(arr, name)
        print(f"  wrote {path}", flush=True)
    nib.save(nib.Nifti1Image(tissue.astype(np.uint8), affine),
             str(out_dir / f"{args.subject}_tissue.nii.gz"))

    # ROI summary
    summary = {"subject": args.subject, "downsample": args.downsample}
    for name, label in [("WM", 1), ("GM", 2), ("CSF", 3), ("CC", 4)]:
        roi_mask = tissue == label
        if not roi_mask.any():
            continue
        for arr_name, arr in [("FA", FA), ("MD", MD), ("MK", MK),
                               ("alpha_persistence", alpha)]:
            vals = arr[roi_mask]
            vals = vals[np.isfinite(vals)]
            summary[f"{name}_{arr_name}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
            summary[f"{name}_{arr_name}_median"] = float(np.median(vals)) if vals.size else float("nan")
            summary[f"{name}_{arr_name}_std"] = float(np.std(vals)) if vals.size else float("nan")
            summary[f"{name}_{arr_name}_n"] = int(vals.size)

    summary_path = out_dir / f"{args.subject}_roi_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {summary_path}", flush=True)
    print(f"[{args.subject}] done in {time.perf_counter() - t0:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
