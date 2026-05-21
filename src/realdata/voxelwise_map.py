"""Voxelwise computation of persistence-tail, kurtosis K, and stretched
alpha maps from a DW-MRI dataset.

The voxelwise persistence-tail estimator uses ``mode='cumulative'`` on
the (1, n_shells, n_dirs) signal of the single voxel. Because each voxel
contributes only one signal, we additionally pool neighbours within a
small spatial window to stabilise the Hill estimator (or set the window
size to 1 to obtain a strictly voxelwise map).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..estimators.dwmri_estimators import (
    PersistenceTailDWMRI,
    KurtosisFit,
    StretchedExponentialFit,
)


def voxelwise_persistence_tail(
    S: NDArray[np.float32],
    pooling_radius: int = 0,
    k_fraction: float = 0.15,
) -> NDArray[np.float32]:
    """Estimate alpha_persistence per voxel from a (n_voxels, n_shells, n_dirs)
    matrix.

    If ``pooling_radius > 0`` the lifetimes from the voxel and its
    nearest ``pooling_radius`` neighbours (in the linear voxel ordering)
    are pooled before applying Hill. This is appropriate only when the
    voxel index reflects a meaningful spatial ordering (e.g. row-major
    indices within a connected ROI).
    """
    n_v = S.shape[0]
    out = np.full(n_v, np.nan, dtype=np.float32)
    estimator = PersistenceTailDWMRI(
        mode="cumulative", k_fraction=k_fraction
    )
    for v in range(n_v):
        a = max(0, v - pooling_radius)
        b = min(n_v, v + pooling_radius + 1)
        block = S[a:b]
        try:
            out[v] = estimator.fit(block)["alpha_hat"]
        except Exception:
            out[v] = np.nan
    return out


def voxelwise_kurtosis(
    S: NDArray[np.float32],
    b_values: NDArray[np.float64],
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Per-voxel DKI fit on direction-averaged signal.

    Returns
    -------
    K : (n_voxels,) array of kurtosis estimates
    D : (n_voxels,) array of diffusivity estimates
    """
    n_v = S.shape[0]
    K = np.full(n_v, np.nan, dtype=np.float32)
    D = np.full(n_v, np.nan, dtype=np.float32)
    fit = KurtosisFit()
    for v in range(n_v):
        s_mean = S[v].mean(axis=-1)
        out = fit.fit(b_values, s_mean)
        K[v] = out["K"]
        D[v] = out["D"]
    return K, D


def voxelwise_stretched(
    S: NDArray[np.float32],
    b_values: NDArray[np.float64],
) -> NDArray[np.float32]:
    """Per-voxel stretched-exponential fit on direction-averaged signal."""
    n_v = S.shape[0]
    out = np.full(n_v, np.nan, dtype=np.float32)
    fit = StretchedExponentialFit()
    for v in range(n_v):
        s_mean = S[v].mean(axis=-1)
        out[v] = fit.fit(b_values, s_mean)["alpha_se"]
    return out


def write_voxel_map_to_nifti(
    values: NDArray[np.float32],
    voxel_indices: NDArray[np.int_],
    reference_affine: NDArray[np.float64],
    reference_shape: Tuple[int, int, int],
    out_path: str,
) -> None:
    """Write a flat (n_voxels,) array back into a 3D NIfTI volume."""
    import nibabel as nib
    volume = np.zeros(reference_shape, dtype=np.float32)
    volume[voxel_indices[:, 0],
           voxel_indices[:, 1],
           voxel_indices[:, 2]] = values
    img = nib.Nifti1Image(volume, reference_affine)
    nib.save(img, str(out_path))
