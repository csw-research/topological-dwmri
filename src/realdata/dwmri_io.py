"""I/O and pre-processing for real DW-MRI data (HCP / MGH / dipy datasets).

The functions in this module assume a standard DW-MRI dataset of:

* ``data.nii.gz``       -- 4D NIfTI of shape (X, Y, Z, n_vols)
* ``bvals``             -- text file with one row of n_vols b-values
* ``bvecs``             -- text file with three rows of n_vols unit vectors
* ``brain_mask.nii.gz`` -- 3D NIfTI binary brain mask

We deliberately do not vendor large NIfTI data: this module provides
loaders that work against externally supplied data on disk.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import nibabel as nib  # type: ignore
except ImportError as e:  # pragma: no cover
    nib = None  # noqa


def load_dwi(
    data_path: str,
    bvals_path: str,
    bvecs_path: str,
    mask_path: Optional[str] = None,
) -> Dict:
    """Load a DW-MRI dataset.

    Returns
    -------
    dict with keys
        ``data``  : (X, Y, Z, n_vols) float32 array
        ``bvals`` : (n_vols,)
        ``bvecs`` : (n_vols, 3)
        ``mask``  : (X, Y, Z) bool array or None
        ``affine``: 4x4 affine matrix
    """
    if nib is None:
        raise ImportError("nibabel is required for load_dwi")
    img = nib.load(str(data_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    affine = img.affine
    bvals = np.loadtxt(bvals_path)
    bvecs = np.loadtxt(bvecs_path)
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T
    mask = None
    if mask_path is not None:
        mimg = nib.load(str(mask_path))
        mask = np.asarray(mimg.dataobj).astype(bool)
    return {
        "data": data,
        "bvals": bvals,
        "bvecs": bvecs,
        "mask": mask,
        "affine": affine,
    }


def shell_average(
    data: NDArray[np.float32],
    bvals: NDArray[np.float64],
    shells: Tuple[float, ...] = (0.0, 1000.0, 2000.0, 3000.0),
    tol: float = 50.0,
) -> Tuple[NDArray[np.float32], NDArray[np.float64]]:
    """Average DW volumes within each b-shell.

    Returns the shell-averaged 4D array of shape (X, Y, Z, n_shells)
    and the matched shell b-values.
    """
    out_shells = []
    out_b = []
    for b0 in shells:
        idx = np.where(np.abs(bvals - b0) <= tol)[0]
        if idx.size == 0:
            continue
        out_shells.append(data[..., idx].mean(axis=-1))
        out_b.append(b0)
    if not out_shells:
        raise ValueError("No volumes matched any shell")
    return np.stack(out_shells, axis=-1), np.asarray(out_b)


def shell_directions(
    bvals: NDArray[np.float64],
    bvecs: NDArray[np.float64],
    shells: Tuple[float, ...] = (0.0, 1000.0, 2000.0, 3000.0),
    tol: float = 50.0,
) -> Dict[float, NDArray[np.float64]]:
    """Return a dict mapping each b-shell to its (n_dir, 3) array of
    unit gradient directions."""
    out = {}
    for b0 in shells:
        idx = np.where(np.abs(bvals - b0) <= tol)[0]
        out[b0] = bvecs[idx]
    return out


def voxel_signal_matrix(
    data: NDArray[np.float32],
    bvals: NDArray[np.float64],
    bvecs: NDArray[np.float64],
    mask: NDArray[np.bool_],
    shells: Tuple[float, ...] = (0.0, 1000.0, 2000.0, 3000.0),
    tol: float = 50.0,
    n_dirs_max: Optional[int] = None,
) -> Dict:
    """Build a (n_voxels, n_shells, n_dirs) signal matrix suitable for
    ``PersistenceTailDWMRI(mode='cumulative')``.

    For each shell (other than b=0) we select up to ``n_dirs_max``
    gradient directions; the b=0 image is replicated across the n_dirs
    axis. Missing shell/voxel data is filled with the b=0 value.
    """
    X, Y, Z, _ = data.shape
    inds = np.argwhere(mask)
    n_v = inds.shape[0]
    n_shells = len(shells)

    # build per-shell index lists
    shell_idx = []
    for b0 in shells:
        idx = np.where(np.abs(bvals - b0) <= tol)[0]
        if n_dirs_max is not None and idx.size > n_dirs_max:
            idx = idx[:n_dirs_max]
        shell_idx.append(idx)
    n_dirs = max(len(idx) for idx in shell_idx[1:])  # ignore b=0 in width

    S = np.zeros((n_v, n_shells, n_dirs), dtype=np.float32)
    used_directions = []
    for k, idx in enumerate(shell_idx):
        for j in range(n_dirs):
            j_use = j if k > 0 else 0
            if j_use >= len(idx):
                # Pad with b=0 image
                vol = data[..., shell_idx[0][0]]
            else:
                vol = data[..., idx[j_use]]
            S[:, k, j] = vol[inds[:, 0], inds[:, 1], inds[:, 2]]
        if k > 0:
            used_directions.append(bvecs[idx])
    return {
        "S": S,
        "voxel_indices": inds,
        "shells": np.asarray(shells),
        "directions_per_shell": used_directions,
    }
