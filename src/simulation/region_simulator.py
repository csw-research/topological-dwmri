"""Region-level simulators: generate a population of voxels with a common
microstructural model and produce an (n_voxels, n_b) attenuation array.

The persistence-tail exponent is estimated from sublevel persistence of
the *attenuation curve* y(b) = -ln(S(b)/S0) across the population of
voxels and across many gradient directions. Pooling over voxels and
directions provides the large lifetime samples needed for the Hill
estimator.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Sequence, Tuple

from ..generators.hardi import (
    simulate_voxel,
    fibonacci_sphere,
    HCP_SHELLS,
)


def simulate_region(
    kind: str,
    n_voxels: int,
    b_values: Sequence[float] = HCP_SHELLS,
    n_dirs: int = 30,
    Delta: float = 0.05,
    n_molecules: int = 5000,
    snr: float = 30.0,
    param_jitter: float = 0.1,
    rng: np.random.Generator = None,
    **params,
) -> Dict[str, NDArray[np.float64]]:
    """Simulate a region of ``n_voxels`` voxels with the same microstructural
    model. Diffusivity / radius / alpha parameters are jittered by a
    fractional ``param_jitter`` standard deviation to mimic biological
    heterogeneity.

    Returns
    -------
    dict with keys
        ``S``           : (n_voxels, n_shells, n_dirs) noisy HARDI signal
        ``S_clean``     : (n_voxels, n_shells, n_dirs) noise-free signal
        ``b_values``    : (n_shells,)
        ``directions``  : (n_dirs, 3)
        ``kind``        : str
        ``params``      : list of dicts (per-voxel parameter draws)
    """
    if rng is None:
        rng = np.random.default_rng()
    b_values = np.asarray(b_values, dtype=float)
    directions = fibonacci_sphere(n_dirs)
    n_shells = b_values.size

    S_clean = np.zeros((n_voxels, n_shells, n_dirs))
    S_noisy = np.zeros((n_voxels, n_shells, n_dirs))
    per_voxel_params: List[Dict] = []

    for v in range(n_voxels):
        jitter = lambda x: x * float(np.exp(rng.normal(0.0, param_jitter)))
        local = {}
        for k, val in params.items():
            if isinstance(val, (int, float, np.integer, np.floating)):
                jv = jitter(val)
                if k == "alpha":
                    jv = float(np.clip(jv, 0.3, 2.0))
                local[k] = jv
            else:
                local[k] = val
        # Use a deterministic per-voxel rng so the simulation is reproducible
        local_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        res = simulate_voxel(
            kind, b_values=b_values, n_dirs=n_dirs, Delta=Delta,
            n_molecules=n_molecules, snr=snr, rng=local_rng, **local,
        )
        # Reuse the canonical direction layout (ignore per-voxel directions
        # returned, which equal the global one anyway).
        S_clean[v] = res["S_clean"]
        S_noisy[v] = res["S_noisy"]
        per_voxel_params.append(local)

    return {
        "S": S_noisy,
        "S_clean": S_clean,
        "b_values": b_values,
        "directions": directions,
        "kind": kind,
        "params": per_voxel_params,
    }


def attenuation_curves(
    S: NDArray[np.float64],
    b_values: NDArray[np.float64],
    eps: float = 1e-6,
) -> NDArray[np.float64]:
    """Convert an (n_voxels, n_shells, n_dirs) signal into per-(voxel,direction)
    attenuation curves of shape (n_voxels * n_dirs, n_shells).

    Each curve is y(b) = -ln(S(b)/S(0)) at a single gradient direction.
    """
    if S.ndim != 3:
        raise ValueError(f"expected S.ndim==3, got {S.ndim}")
    n_v, n_s, n_d = S.shape
    # S(0) is the b=0 image (assumed to be the first shell)
    S0 = S[:, 0:1, :]
    S0 = np.maximum(S0, eps)
    ratio = np.maximum(S / S0, eps)
    y = -np.log(ratio)  # shape (n_v, n_s, n_d)
    # rearrange to (n_v * n_d, n_s)
    y = np.transpose(y, (0, 2, 1)).reshape(n_v * n_d, n_s)
    return y


def voxelwise_signal_curves(
    S: NDArray[np.float64],
    average_directions: bool = True,
    eps: float = 1e-6,
) -> NDArray[np.float64]:
    """Return per-voxel S(b) curves, averaged across directions if requested.

    Output shape: (n_voxels, n_shells).
    """
    if S.ndim != 3:
        raise ValueError(f"expected S.ndim==3, got {S.ndim}")
    if average_directions:
        return S.mean(axis=2)
    return S.reshape(S.shape[0] * S.shape[2], S.shape[1])
