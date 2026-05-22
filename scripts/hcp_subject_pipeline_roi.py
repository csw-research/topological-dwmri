#!/usr/bin/env python3
"""HCP per-subject ROI-level alpha_persistence pipeline.

For each subject and each tissue ROI (WM, GM, CSF, CC and optional JHU
labels), we:

  1. Load all per-direction attenuation profiles y_{v,k,i} = -ln(S/S0)
     for the voxels v inside the ROI, across all shells k and directions i.
  2. Concatenate them into a single 1D walk:
        Z_ROI = cumulative sum (over voxels and directions, per shell)
                 of the centred attenuations,
     yielding a Levy-bridge-like sample path whose length is
     n_voxels x n_shells x n_dirs.
  3. Compute sublevel persistence and the Hill estimator on the lifetimes.

The result is ONE alpha_persistence value per (ROI, subject, scan).

Compared with the voxelwise pipeline this:
  * sacrifices voxelwise resolution,
  * destroys arbitrary direction ordering by averaging over voxels,
  * inherits the parent paper Theorem 1 because aggregating alpha-stable
    increments across voxels in a homogeneous ROI yields another
    alpha-stable process with the same alpha,
  * is expected to produce per-subject scan-rescan ICC in the 0.7-0.9
    range based on the dMRI test-retest literature.

This script reuses the v2 pipeline's DTI/DKI fits and tissue
segmentation; only the alpha_persistence computation changes.
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
# Re-use helpers from the v2 pipeline
# ---------------------------------------------------------------------------

from scripts.hcp_subject_pipeline_v2 import (
    fit_dti, fit_dki, segment_tissue, corpus_callosum_mask,
    register_jhu_atlas,
)


# ---------------------------------------------------------------------------
# ROI-level alpha_persistence
# ---------------------------------------------------------------------------

def roi_alpha_persistence(
    data: np.ndarray,
    roi_mask: np.ndarray,
    bvals: np.ndarray,
    shells=(1000.0, 2000.0, 3000.0),
    tol: float = 50.0,
    k_fraction: float = 0.10,
    min_k: int = 20,
):
    """Pool per-(voxel, direction) attenuation curves across the ROI;
    construct one alpha-stable bridge per shell, pool lifetimes across
    shells, apply Hill estimator.

    Returns a dict {alpha_hat, n_lifetimes, k_used, n_voxels}.
    """
    b0_idx = np.where(bvals <= tol)[0]
    if b0_idx.size == 0:
        return {"alpha_hat": np.nan, "n_lifetimes": 0,
                "k_used": 0, "n_voxels": 0}
    b0_mean = data[..., b0_idx].mean(axis=-1).astype(np.float32)
    b0_safe = np.maximum(b0_mean, 1.0)
    n_vox = int(roi_mask.sum())
    if n_vox == 0:
        return {"alpha_hat": np.nan, "n_lifetimes": 0,
                "k_used": 0, "n_voxels": 0}
    inds = np.argwhere(roi_mask)

    all_lifetimes = []
    for s in shells:
        idx = np.where(np.abs(bvals - s) <= tol)[0]
        if idx.size < 4:
            continue
        # Attenuation: shape (n_voxels, n_dirs)
        shell_vols = data[..., idx][inds[:, 0], inds[:, 1], inds[:, 2]]
        y = -np.log(np.maximum(shell_vols / b0_safe[inds[:, 0],
                                                      inds[:, 1],
                                                      inds[:, 2]][:, None],
                                1e-6))
        if not np.all(np.isfinite(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        # Centre across the (voxel, direction) ensemble
        y = y - y.mean()
        # Build the walk by row-major concatenation, then cumulative sum
        flat = y.ravel(order="C")
        Z = np.cumsum(flat)
        diag = sublevel_persistence_1d(Z)
        life = persistence_lifetimes(diag)
        if life.size > 0:
            all_lifetimes.append(life)

    if not all_lifetimes:
        return {"alpha_hat": np.nan, "n_lifetimes": 0,
                "k_used": 0, "n_voxels": n_vox}

    L = np.sort(np.concatenate(all_lifetimes))[::-1]
    N = L.size
    if N < min_k + 1:
        return {"alpha_hat": np.nan, "n_lifetimes": N,
                "k_used": 0, "n_voxels": n_vox}
    k = max(min_k, int(k_fraction * N))
    k = min(k, N - 1)
    log_top = np.log(L[:k]); log_thresh = np.log(L[k])
    m = float(np.mean(log_top - log_thresh))
    if m <= 0:
        return {"alpha_hat": np.nan, "n_lifetimes": N,
                "k_used": k, "n_voxels": n_vox}
    return {"alpha_hat": 1.0 / m, "n_lifetimes": N, "k_used": k,
            "n_voxels": n_vox}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--bvals", required=True)
    p.add_argument("--bvecs", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--downsample", type=int, default=2)
    p.add_argument("--denoise", action="store_true")
    p.add_argument("--skip-jhu", action="store_true")
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

    if args.downsample > 1:
        s = args.downsample
        data = data[::s, ::s, ::s]
        mask = mask[::s, ::s, ::s]
        scale = np.eye(4); scale[:3, :3] *= s
        affine = affine @ scale
        print(f"  downsampled to {data.shape}", flush=True)

    if args.denoise:
        print(f"[{args.subject}] MP-PCA denoising...", flush=True)
        from dipy.denoise.localpca import mppca
        data = mppca(data, patch_radius=2, mask=mask).astype(np.float32)
        print("  denoised", flush=True)

    print(f"[{args.subject}] DTI...", flush=True)
    FA, MD = fit_dti(data, mask, bvals, bvecs)
    print(f"[{args.subject}] DKI...", flush=True)
    try:
        MK = fit_dki(data, mask, bvals, bvecs)
    except Exception as e:
        print(f"  DKI failed: {e!r}"); MK = np.zeros_like(FA)

    print(f"[{args.subject}] tissue...", flush=True)
    b0_idx = np.where(bvals <= 50)[0]
    b0_mean = data[..., b0_idx].mean(axis=-1)
    tissue = segment_tissue(FA, b0_mean, mask)
    tissue[corpus_callosum_mask(tissue, FA)] = 4

    summary = {"subject": args.subject, "downsample": args.downsample,
                "denoise": bool(args.denoise)}

    def add_roi(name, m3):
        if not m3.any():
            return
        r = roi_alpha_persistence(data, m3, bvals)
        # FA / MD / MK ROI means
        for arr_name, arr in [("FA", FA), ("MD", MD), ("MK", MK)]:
            vals = arr[m3]
            vals = vals[np.isfinite(vals)]
            summary[f"{name}_{arr_name}"] = (float(np.mean(vals))
                                              if vals.size else float("nan"))
        summary[f"{name}_alpha_persistence"] = float(r["alpha_hat"])
        summary[f"{name}_n_lifetimes"] = int(r["n_lifetimes"])
        summary[f"{name}_n_voxels"] = int(r["n_voxels"])

    print(f"[{args.subject}] ROI alpha_persistence...", flush=True)
    for name, lbl in [("WM", 1), ("GM", 2), ("CSF", 3), ("CC", 4)]:
        m3 = tissue == lbl
        add_roi(name, m3)
        print(f"  {name}: alpha={summary[f'{name}_alpha_persistence']:.3f} "
              f"(n_vox={summary[f'{name}_n_voxels']}, "
              f"n_life={summary[f'{name}_n_lifetimes']})", flush=True)

    if not args.skip_jhu:
        print(f"[{args.subject}] JHU registration...", flush=True)
        try:
            jhu_labels, jhu_names = register_jhu_atlas(
                FA, affine, out_dir / "_flirt_work"
            )
            summary["jhu_labels"] = jhu_names
            ids, counts = np.unique(jhu_labels, return_counts=True)
            for lid, cnt in zip(ids.tolist(), counts.tolist()):
                if lid == 0 or cnt < 50:
                    continue
                name = jhu_names.get(lid, f"jhu_{lid}").replace(" ", "_") \
                                                       .replace("/", "_")
                add_roi(f"JHU_{lid:02d}_{name}", jhu_labels == lid)
        except Exception as e:
            print(f"  JHU failed: {e!r}", flush=True)

    path = out_dir / f"{args.subject}_roi_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {path}", flush=True)

    import shutil
    flirt_work = out_dir / "_flirt_work"
    if flirt_work.is_dir():
        shutil.rmtree(flirt_work, ignore_errors=True)
    print(f"[{args.subject}] done in {time.perf_counter()-t0:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
