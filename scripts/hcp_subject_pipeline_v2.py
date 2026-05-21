#!/usr/bin/env python3
"""HCP per-subject pipeline v2 — adds JHU ICBM-DTI-81 atlas ROIs and
split-half test-retest.

For each subject:

  1. Load data / bvals / bvecs / brain mask.
  2. Fit DTI (FA, MD), DKI (MK).
  3. Coarse WM / GM / CSF / midsagittal CC mask.
  4. Voxelwise alpha_persistence (cumulative-bridge construction with
     6-connected spatial neighbourhood pooling).
  5. Split-half: redo step 4 on two interleaved halves of the gradient
     directions (per shell) to obtain alpha_persistence_split_A and
     alpha_persistence_split_B (test-retest analogue).
  6. JHU atlas:
       (a) FSL flirt: subject FA -> FMRIB58 FA template (affine, 12 dof)
       (b) Inverse-warp the JHU-ICBM labels (2 mm) back to subject space
           with nearest-neighbour interpolation.
       (c) For each labelled region (top-N by voxel count) compute mean,
           median, std of FA, MD, MK, alpha_persistence,
           alpha_persistence_split_A, alpha_persistence_split_B.
  7. Save:
        <subject>_FA.nii.gz / MD / MK / alpha_persistence
        <subject>_alpha_split_A.nii.gz, <subject>_alpha_split_B.nii.gz
        <subject>_tissue.nii.gz, <subject>_jhu_atlas.nii.gz
        <subject>_roi_summary.json (tissue + per-JHU-label)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
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

# --- defaults --------------------------------------------------------------
FSL_DIR = Path("/share/software/user/open/fsl/5.0.10")
FMRIB58_FA = FSL_DIR / "data/standard/FMRIB58_FA_1mm.nii.gz"
JHU_LABELS = FSL_DIR / "data/atlases/JHU/JHU-ICBM-labels-2mm.nii.gz"
JHU_LABELS_XML = FSL_DIR / "data/atlases/JHU-labels.xml"


# ---------------------------------------------------------------------------
# DTI / DKI fits via dipy
# ---------------------------------------------------------------------------

def fit_dti(data, mask, bvals, bvecs):
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)
    fit = TensorModel(gtab).fit(data, mask=mask)
    return fit.fa.astype(np.float32), fit.md.astype(np.float32)


def fit_dki(data, mask, bvals, bvecs):
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dki import DiffusionKurtosisModel
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)
    fit = DiffusionKurtosisModel(gtab).fit(data, mask=mask)
    return fit.mk(min_kurtosis=-0.5, max_kurtosis=3.0).astype(np.float32)


def segment_tissue(FA, mean_b0, mask):
    out = np.zeros_like(FA, dtype=np.uint8)
    b0_norm = mean_b0 / (np.percentile(mean_b0[mask], 99) + 1e-6)
    csf = mask & (b0_norm > 0.8) & (FA < 0.15)
    wm = mask & (FA >= 0.4) & ~csf
    gm = mask & ~csf & ~wm
    out[wm] = 1; out[gm] = 2; out[csf] = 3
    return out


def corpus_callosum_mask(tissue, FA):
    cc = np.zeros_like(tissue, dtype=bool)
    midx = tissue.shape[0] // 2
    band = slice(midx - 2, midx + 3)
    cc[band] = (tissue[band] == 1) & (FA[band] >= 0.6)
    return cc


# ---------------------------------------------------------------------------
# Voxelwise alpha_persistence (with optional direction subsetting)
# ---------------------------------------------------------------------------

def voxelwise_alpha_persistence(
    data, mask, bvals, bvecs,
    shells=(1000.0, 2000.0, 3000.0),
    tol=50.0,
    pooling_radius=1,
    k_fraction=0.15,
    min_k=8,
    direction_subset="full",   # "full", "A", "B"
):
    b0_idx = np.where(bvals <= tol)[0]
    if b0_idx.size == 0:
        raise RuntimeError("No b=0 volumes")
    b0_mean = data[..., b0_idx].mean(axis=-1).astype(np.float32)
    b0_safe = np.maximum(b0_mean, 1.0)

    shell_idx = {}
    for s in shells:
        idx = np.where(np.abs(bvals - s) <= tol)[0]
        if direction_subset == "A":
            idx = idx[0::2]
        elif direction_subset == "B":
            idx = idx[1::2]
        if idx.size >= 4:
            shell_idx[s] = idx

    inds = np.argwhere(mask)
    out = np.full(mask.shape, np.nan, dtype=np.float32)
    per_voxel_lifetimes = [[] for _ in range(inds.shape[0])]
    for s, idx in shell_idx.items():
        shell_vols = data[..., idx]
        attn = -np.log(np.maximum(shell_vols / b0_safe[..., None], 1e-6))
        for k, (xi, yi, zi) in enumerate(inds):
            y = attn[xi, yi, zi]
            if not np.all(np.isfinite(y)):
                continue
            y = y - y.mean()
            Zk = np.cumsum(y)
            life = persistence_lifetimes(sublevel_persistence_1d(Zk))
            if life.size > 0:
                per_voxel_lifetimes[k].append(life)

    if pooling_radius > 0:
        loc = {tuple(inds[k]): k for k in range(inds.shape[0])}
        rad = pooling_radius
        for k, (xi, yi, zi) in enumerate(inds):
            collected = list(per_voxel_lifetimes[k])
            for dx in range(-rad, rad + 1):
                for dy in range(-rad, rad + 1):
                    for dz in range(-rad, rad + 1):
                        if dx == dy == dz == 0:
                            continue
                        j = loc.get((xi + dx, yi + dy, zi + dz))
                        if j is not None:
                            collected.extend(per_voxel_lifetimes[j])
            if not collected:
                continue
            L = np.sort(np.concatenate(collected))[::-1]
            if L.size < min_k + 1:
                continue
            kk = min(max(min_k, int(k_fraction * L.size)), L.size - 1)
            m = float(np.mean(np.log(L[:kk]) - np.log(L[kk])))
            if m > 0:
                out[xi, yi, zi] = 1.0 / m
    else:
        for k, (xi, yi, zi) in enumerate(inds):
            if not per_voxel_lifetimes[k]:
                continue
            L = np.sort(np.concatenate(per_voxel_lifetimes[k]))[::-1]
            if L.size < min_k + 1:
                continue
            kk = min(max(min_k, int(k_fraction * L.size)), L.size - 1)
            m = float(np.mean(np.log(L[:kk]) - np.log(L[kk])))
            if m > 0:
                out[xi, yi, zi] = 1.0 / m
    return out


# ---------------------------------------------------------------------------
# JHU atlas registration
# ---------------------------------------------------------------------------

def register_jhu_atlas(FA, affine, work_dir):
    """Register subject FA to FMRIB58 FA template and warp JHU labels back.

    Uses FSL flirt (affine, 12-dof). Returns a JHU label volume in subject
    space (same shape as FA) and a list of (label_id, label_name).
    """
    work_dir = Path(work_dir); work_dir.mkdir(parents=True, exist_ok=True)
    fa_path = work_dir / "subj_FA.nii.gz"
    nib.save(nib.Nifti1Image(FA.astype(np.float32), affine), str(fa_path))

    sub2mni = work_dir / "subj2mni.mat"
    mni2sub = work_dir / "mni2subj.mat"
    fa_in_mni = work_dir / "subj_FA_in_mni.nii.gz"
    atlas_in_sub = work_dir / "jhu_in_subj.nii.gz"

    env = {**__import__("os").environ, "FSLDIR": str(FSL_DIR),
           "PATH": f"{FSL_DIR}/bin:" + __import__("os").environ.get("PATH", "")}

    cmd1 = [str(FSL_DIR / "bin/flirt"),
            "-in", str(fa_path),
            "-ref", str(FMRIB58_FA),
            "-omat", str(sub2mni),
            "-out", str(fa_in_mni),
            "-dof", "12"]
    subprocess.run(cmd1, check=True, env=env,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd2 = [str(FSL_DIR / "bin/convert_xfm"),
            "-omat", str(mni2sub),
            "-inverse", str(sub2mni)]
    subprocess.run(cmd2, check=True, env=env)
    cmd3 = [str(FSL_DIR / "bin/flirt"),
            "-in", str(JHU_LABELS),
            "-ref", str(fa_path),
            "-applyxfm",
            "-init", str(mni2sub),
            "-interp", "nearestneighbour",
            "-out", str(atlas_in_sub)]
    subprocess.run(cmd3, check=True, env=env,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    labels = nib.load(str(atlas_in_sub)).get_fdata().astype(np.int32)

    # Parse JHU label names from the XML
    try:
        import xml.etree.ElementTree as ET
        root = ET.parse(str(JHU_LABELS_XML)).getroot()
        names = {}
        for L in root.findall(".//label"):
            idx = int(L.attrib["index"])
            names[idx + 1] = L.text  # XML index is 0-based, atlas is 1-based
    except Exception:
        names = {i: f"label_{i}" for i in range(1, 49)}
    return labels, names


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
    p.add_argument("--pooling-radius", type=int, default=1)
    p.add_argument("--downsample", type=int, default=2)
    p.add_argument("--skip-jhu", action="store_true",
                   help="Skip JHU atlas registration (FSL not available).")
    p.add_argument("--skip-splits", action="store_true",
                   help="Skip split-half test-retest computation.")
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

    print(f"[{args.subject}] DTI fit...", flush=True)
    FA, MD = fit_dti(data, mask, bvals, bvecs)
    print(f"[{args.subject}] DKI fit...", flush=True)
    try:
        MK = fit_dki(data, mask, bvals, bvecs)
    except Exception as e:
        print(f"  DKI failed: {e!r}; using zeros")
        MK = np.zeros_like(FA)

    print(f"[{args.subject}] tissue segmentation...", flush=True)
    b0_idx = np.where(bvals <= 50)[0]
    b0_mean = data[..., b0_idx].mean(axis=-1)
    tissue = segment_tissue(FA, b0_mean, mask)
    tissue[corpus_callosum_mask(tissue, FA)] = 4
    print(f"  WM/GM/CSF/CC: "
          f"{(tissue==1).sum()}, {(tissue==2).sum()}, "
          f"{(tissue==3).sum()}, {(tissue==4).sum()}", flush=True)

    print(f"[{args.subject}] voxelwise alpha_persistence (full)...",
          flush=True)
    alpha = voxelwise_alpha_persistence(
        data, mask, bvals, bvecs,
        pooling_radius=args.pooling_radius, direction_subset="full",
    )

    alpha_A = alpha_B = None
    if not args.skip_splits:
        print(f"[{args.subject}] split-half A...", flush=True)
        alpha_A = voxelwise_alpha_persistence(
            data, mask, bvals, bvecs,
            pooling_radius=args.pooling_radius, direction_subset="A",
        )
        print(f"[{args.subject}] split-half B...", flush=True)
        alpha_B = voxelwise_alpha_persistence(
            data, mask, bvals, bvecs,
            pooling_radius=args.pooling_radius, direction_subset="B",
        )

    jhu_labels = None; jhu_names = None
    if not args.skip_jhu:
        print(f"[{args.subject}] JHU atlas registration...", flush=True)
        try:
            jhu_labels, jhu_names = register_jhu_atlas(
                FA, affine, out_dir / "_flirt_work"
            )
            print(f"  JHU labels found: {len(np.unique(jhu_labels))} "
                  f"unique values", flush=True)
        except Exception as e:
            print(f"  JHU registration failed: {e!r}", flush=True)
            jhu_labels = None

    # --- save NIfTIs ------------------------------------------------------
    def save(arr, name, dtype=np.float32):
        path = out_dir / f"{args.subject}_{name}.nii.gz"
        nib.save(nib.Nifti1Image(arr.astype(dtype), affine), str(path))
        return path
    save(FA, "FA"); save(MD, "MD"); save(MK, "MK")
    save(alpha, "alpha_persistence")
    if alpha_A is not None:
        save(alpha_A, "alpha_split_A")
        save(alpha_B, "alpha_split_B")
    save(tissue, "tissue", dtype=np.uint8)
    if jhu_labels is not None:
        save(jhu_labels, "jhu_atlas", dtype=np.int16)

    # --- summary ----------------------------------------------------------
    summary = {"subject": args.subject, "downsample": args.downsample}

    def add_roi(name, mask3d, maps):
        for arr_name, arr in maps:
            vals = arr[mask3d]
            vals = vals[np.isfinite(vals)]
            summary[f"{name}_{arr_name}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
            summary[f"{name}_{arr_name}_median"] = float(np.median(vals)) if vals.size else float("nan")
            summary[f"{name}_{arr_name}_std"] = float(np.std(vals)) if vals.size else float("nan")
            summary[f"{name}_{arr_name}_n"] = int(vals.size)

    base_maps = [("FA", FA), ("MD", MD), ("MK", MK),
                 ("alpha_persistence", alpha)]
    if alpha_A is not None:
        base_maps += [("alpha_split_A", alpha_A),
                       ("alpha_split_B", alpha_B)]

    for name, lbl in [("WM", 1), ("GM", 2), ("CSF", 3), ("CC", 4)]:
        m3 = tissue == lbl
        if m3.any():
            add_roi(name, m3, base_maps)

    if jhu_labels is not None:
        summary["jhu_labels"] = jhu_names
        ids, counts = np.unique(jhu_labels, return_counts=True)
        # Skip background (0); include all labels with >=50 voxels
        for lid, cnt in zip(ids.tolist(), counts.tolist()):
            if lid == 0 or cnt < 50:
                continue
            name = jhu_names.get(lid, f"jhu_{lid}").replace(" ", "_") \
                                                  .replace("/", "_")
            add_roi(f"JHU_{lid:02d}_{name}", jhu_labels == lid, base_maps)

    path = out_dir / f"{args.subject}_roi_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {path}", flush=True)

    # Clean up flirt working dir
    flirt_work = out_dir / "_flirt_work"
    if flirt_work.is_dir():
        shutil.rmtree(flirt_work, ignore_errors=True)

    print(f"[{args.subject}] done in {time.perf_counter()-t0:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
