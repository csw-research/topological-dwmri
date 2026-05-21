"""High-angular-resolution DW-MRI (HARDI) signal simulation.

For each voxel we generate a 3D displacement sample (one row per molecule),
then evaluate the empirical signal at a set of (b, g_hat) acquisition
directions. The output is an array ``S[shell, direction]`` matching the
input expected by ``PersistenceTailDWMRI(mode='directional')``.

Three voxel kinds are supported:

* ``'free'``       -- isotropic Gaussian diffusion
* ``'restricted'`` -- anisotropic Gaussian with one perpendicular axis
                       saturating to a restricted MSD (cylinder model)
* ``'stable'``     -- isotropic symmetric alpha-stable displacements

The HCP-style shell schedule is exposed as ``HCP_SHELLS``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Sequence, Tuple

from .dwmri_signal import sample_stable_displacement


# ---------------------------------------------------------------------------
# Direction sampling on the unit sphere
# ---------------------------------------------------------------------------

def fibonacci_sphere(n: int) -> NDArray[np.float64]:
    """Return n approximately equidistributed unit vectors on the sphere."""
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * indices
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def random_sphere(n: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Return n uniform random unit vectors on the sphere."""
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


# ---------------------------------------------------------------------------
# Voxel displacement generators
# ---------------------------------------------------------------------------

def displacements_free(
    D: float,
    Delta: float,
    n_molecules: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Isotropic Gaussian displacements (per-axis var = 2 D Delta)."""
    return rng.normal(0.0, np.sqrt(2.0 * D * Delta), size=(n_molecules, 3))


def displacements_restricted(
    D_perp: float,
    D_par: float,
    Delta: float,
    n_molecules: int,
    fibre_dir: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Anisotropic Gaussian displacements with given parallel/perpendicular
    diffusivities. ``fibre_dir`` is a unit vector defining the parallel axis.

    Suitable as a long-Delta limit of cylinder restriction.
    """
    sigma_par = np.sqrt(2.0 * D_par * Delta)
    sigma_perp = np.sqrt(2.0 * D_perp * Delta)
    fibre_dir = fibre_dir / np.linalg.norm(fibre_dir)
    # local orthonormal basis
    a = np.array([1.0, 0.0, 0.0])
    if abs(fibre_dir @ a) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(fibre_dir, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(fibre_dir, e1)
    u = sigma_par * rng.normal(size=n_molecules)
    v = sigma_perp * rng.normal(size=n_molecules)
    w = sigma_perp * rng.normal(size=n_molecules)
    return np.outer(u, fibre_dir) + np.outer(v, e1) + np.outer(w, e2)


def displacements_stable(
    alpha: float,
    D_alpha: float,
    Delta: float,
    n_molecules: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Isotropic symmetric alpha-stable displacements (3D).

    A 3D isotropic symmetric alpha-stable vector with scale c has
    characteristic function exp(-c^alpha |q|^alpha) and may be sampled as
    R u where R has the radial alpha-stable subordinator marginal and u is
    uniform on the sphere. We use the equivalent representation
    X = sqrt(T) G, where G is standard 3D Gaussian and T is a positive
    (alpha/2)-stable subordinator (Veillette, Taqqu 2010).
    """
    if not 0 < alpha <= 2:
        raise ValueError(f"alpha must be in (0, 2], got {alpha}")
    if alpha == 2.0:
        return rng.normal(0.0, np.sqrt(2.0 * D_alpha * Delta),
                          size=(n_molecules, 3))
    # Positive (alpha/2)-stable subordinator T with Laplace transform
    # exp(-c_T s^{alpha/2}). Sample as totally-skewed stable with beta=1.
    # We use the parent paper's stable_rvs in 0-parameterization. After
    # combining with the Gaussian factor sqrt(T) G, the resulting vector
    # has characteristic function exp(-c^alpha |q|^alpha) with
    # c = (D_alpha Delta)^(1/2).
    from ..utils.parent_bridge import stable_rvs
    half_alpha = alpha / 2.0
    # 0-parameterised totally skewed positive stable subordinator scale
    # such that E[exp(-s T)] = exp(-(s)^(alpha/2)).
    T = stable_rvs(alpha=half_alpha, beta=1.0, scale=1.0, size=n_molecules,
                   rng=rng)
    T = np.maximum(T, 0.0)
    G = rng.normal(size=(n_molecules, 3))
    c = (D_alpha * Delta) ** 0.5
    return c * np.sqrt(T)[:, None] * G


# ---------------------------------------------------------------------------
# HARDI signal evaluation
# ---------------------------------------------------------------------------

def hardi_signal(
    displacements: NDArray[np.float64],
    b_values: Sequence[float],
    directions: NDArray[np.float64],
    Delta: float,
    S0: float = 1.0,
) -> NDArray[np.float64]:
    """Empirical HARDI signal S[shell, direction].

    For a 3D displacement vector r and acquisition (b, g_hat),
        S(b, g_hat) = E[exp(i q g_hat . r)] = E[cos(q g_hat . r)]
    where q = sqrt(b / Delta). We average over the molecules.
    """
    b_values = np.asarray(b_values, dtype=float)
    directions = np.asarray(directions, dtype=float)
    n_shells = b_values.size
    n_dirs = directions.shape[0]
    out = np.zeros((n_shells, n_dirs))
    proj = displacements @ directions.T  # shape (n_mol, n_dirs)
    for k, b in enumerate(b_values):
        q = np.sqrt(max(b, 0.0) / max(Delta, 1e-12))
        out[k] = S0 * np.mean(np.cos(q * proj), axis=0)
    return out


# ---------------------------------------------------------------------------
# Voxel-level convenience wrapper
# ---------------------------------------------------------------------------

#: HCP-style shell schedule (s/mm^2).
HCP_SHELLS: Tuple[float, ...] = (0.0, 1000.0, 2000.0, 3000.0)


def simulate_voxel(
    kind: str,
    b_values: Sequence[float] = HCP_SHELLS,
    n_dirs: int = 90,
    Delta: float = 0.05,
    n_molecules: int = 20000,
    snr: float = 30.0,
    rng: np.random.Generator = None,
    **params,
) -> Dict[str, NDArray[np.float64]]:
    """Simulate a single voxel and return the noise-free + noisy HARDI signal.

    ``kind`` selects the displacement model; remaining keyword arguments
    are forwarded to that model:

    * ``'free'``        -- requires ``D``
    * ``'restricted'``  -- requires ``D_perp``, ``D_par``, ``fibre_dir``
    * ``'stable'``      -- requires ``alpha`` and ``D_alpha``
    * ``'mixture'``     -- requires ``compartments``, a list of dicts
                            each with ``fraction`` and the keys above

    Returns a dict with keys ``S_clean``, ``S_noisy``, ``directions``,
    ``b_values``.
    """
    if rng is None:
        rng = np.random.default_rng()
    directions = fibonacci_sphere(n_dirs)

    if kind == "free":
        disp = displacements_free(params["D"], Delta, n_molecules, rng)
    elif kind == "restricted":
        fibre_dir = np.asarray(params.get("fibre_dir", [0.0, 0.0, 1.0]),
                               dtype=float)
        disp = displacements_restricted(
            params["D_perp"], params["D_par"], Delta, n_molecules,
            fibre_dir, rng,
        )
    elif kind == "stable":
        disp = displacements_stable(
            params["alpha"], params["D_alpha"], Delta, n_molecules, rng,
        )
    elif kind == "mixture":
        comps = params["compartments"]
        # Allocate molecules to compartments by fraction
        ns = [int(round(c["fraction"] * n_molecules)) for c in comps]
        ns[-1] = n_molecules - sum(ns[:-1])
        parts = []
        for c, n_c in zip(comps, ns):
            sub_kind = c["kind"]
            if sub_kind == "free":
                parts.append(displacements_free(c["D"], Delta, n_c, rng))
            elif sub_kind == "restricted":
                fd = np.asarray(c.get("fibre_dir", [0, 0, 1]), dtype=float)
                parts.append(displacements_restricted(
                    c["D_perp"], c["D_par"], Delta, n_c, fd, rng,
                ))
            elif sub_kind == "stable":
                parts.append(displacements_stable(
                    c["alpha"], c["D_alpha"], Delta, n_c, rng,
                ))
            else:
                raise ValueError(f"Unknown sub_kind={sub_kind}")
        disp = np.concatenate(parts, axis=0)
    else:
        raise ValueError(f"Unknown voxel kind: {kind}")

    S_clean = hardi_signal(disp, b_values, directions, Delta=Delta)

    # Rician noise
    sigma = 1.0 / snr
    nr = rng.normal(0.0, sigma, size=S_clean.shape)
    ni = rng.normal(0.0, sigma, size=S_clean.shape)
    S_noisy = np.sqrt((S_clean + nr) ** 2 + ni ** 2)

    return {
        "S_clean": S_clean,
        "S_noisy": S_noisy,
        "directions": directions,
        "b_values": np.asarray(b_values, dtype=float),
    }
