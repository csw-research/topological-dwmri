#!/usr/bin/env python3
"""Per-subject HCP/MGH DW-MRI processing.

Loads a DW-MRI volume + bvals/bvecs/mask, computes voxelwise
persistence-tail, kurtosis, and stretched-alpha maps, and writes them
back to NIfTI files.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.realdata.dwmri_io import load_dwi, voxel_signal_matrix
from src.realdata.voxelwise_map import (
    voxelwise_persistence_tail,
    voxelwise_kurtosis,
    voxelwise_stretched,
    write_voxel_map_to_nifti,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True)
    p.add_argument("--data",    required=True)
    p.add_argument("--bvals",   required=True)
    p.add_argument("--bvecs",   required=True)
    p.add_argument("--mask",    required=True)
    p.add_argument("--out",     required=True)
    p.add_argument("--shells",  default="0,1000,2000,3000",
                   help="Comma-separated b-shells to include.")
    p.add_argument("--n-dirs-max", type=int, default=90)
    p.add_argument("--pooling-radius", type=int, default=2,
                   help="Voxel neighbourhood radius (linear index) for "
                        "persistence pooling.")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shells = tuple(float(s) for s in args.shells.split(","))

    t0 = time.perf_counter()
    print(f"[{args.subject}] loading DW-MRI...", flush=True)
    dwi = load_dwi(args.data, args.bvals, args.bvecs, args.mask)
    print(f"  data shape   : {dwi['data'].shape}")
    print(f"  bvals unique : {np.unique(dwi['bvals'].round())}")
    print(f"  mask voxels  : {dwi['mask'].sum() if dwi['mask'] is not None else 'no mask'}")

    print(f"[{args.subject}] building voxel signal matrix...", flush=True)
    matrix = voxel_signal_matrix(
        dwi["data"], dwi["bvals"], dwi["bvecs"],
        mask=dwi["mask"],
        shells=shells, tol=50.0,
        n_dirs_max=args.n_dirs_max,
    )
    S = matrix["S"]
    inds = matrix["voxel_indices"]
    b_arr = matrix["shells"]
    print(f"  S shape      : {S.shape}", flush=True)

    print(f"[{args.subject}] computing persistence-tail map...", flush=True)
    alpha_map = voxelwise_persistence_tail(
        S, pooling_radius=args.pooling_radius
    )
    print(f"[{args.subject}] computing DKI map...", flush=True)
    K_map, D_map = voxelwise_kurtosis(S, b_arr)
    print(f"[{args.subject}] computing stretched-alpha map...", flush=True)
    se_map = voxelwise_stretched(S, b_arr)

    ref_shape = dwi["data"].shape[:3]
    for name, vals in [
        ("alpha_persistence", alpha_map),
        ("K", K_map),
        ("D", D_map),
        ("alpha_se", se_map),
    ]:
        path = out_dir / f"{args.subject}_{name}.nii.gz"
        write_voxel_map_to_nifti(vals, inds, dwi["affine"], ref_shape, path)
        print(f"  wrote {path}", flush=True)

    print(f"[{args.subject}] done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
