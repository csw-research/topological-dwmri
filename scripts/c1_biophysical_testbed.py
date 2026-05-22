#!/usr/bin/env python3
"""C1 biophysical simulation testbed (interpretation-1 retry).

The previous testbed (c1_simulation_testbed.py) tested bridges built
from increment distributions. The CLT washed out any difference
between bounded-support and Gaussian increments at the level of the
summed walk, so the test was uninformative.

This testbed corrects that. The simulation now operates at the
biophysical signal level:

1. Simulate a population of voxels with a known microstructure
   (free Gaussian diffusion, restricted cylinders/spheres, alpha-stable
    displacement, or a multi-compartment mixture).
2. From each voxel's molecular displacement distribution, evaluate
   the DW-MRI signal S(b, g_hat) at HCP-like (b, g_hat) sampling.
3. Build the per-(voxel, shell, direction) attenuation
   y = -ln(S/S_0), pool across the ROI, centre per shell, and
   cumulatively sum into the discrete Levy-bridge approximation
   used by the in-vivo pipeline.
4. Compute sublevel persistence and lifetimes.
5. Evaluate the Hill estimator, genpareto shape xi, and the
   Wasserstein-to-Brownian-reference distance W1.

The honest test for C1: do xi or W1 separate restricted from free at
the Gaussian boundary in this biophysically faithful setup?
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.parent_bridge import (
    sublevel_persistence_1d,
    persistence_lifetimes,
)
from src.generators.hardi import (
    displacements_free, displacements_restricted, displacements_stable,
    hardi_signal, fibonacci_sphere, HCP_SHELLS,
)
from src.estimators.boundary_estimators import (
    GenParetoShapeEstimator, WassersteinAnomalyEstimator,
)


# ---------------------------------------------------------------------------
# Build the L(b) walk from a simulated ROI
# ---------------------------------------------------------------------------

def build_roi_walk_and_lifetimes(
    S: np.ndarray, b_values: np.ndarray, eps: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """Given an (n_voxels, n_shells, n_dirs) HARDI signal, build the
    ROI-level Levy-bridge walk exactly as the in-vivo pipeline does,
    and return the sublevel-persistence lifetimes.

    Output: dict with 'lifetimes' (pooled across shells) and 'walks'
    (one numpy array per shell). The path length is reported so that
    the W1 reference can be matched.
    """
    n_v, n_s, n_d = S.shape
    S0 = np.maximum(S[:, 0:1, :], eps)  # use first shell as b=0 ref
    ratio = np.clip(S / S0, eps, None)
    y = -np.log(ratio)
    pooled_life = []
    walks_per_shell = []
    for k in range(1, n_s):  # skip the b=0 shell
        flat = y[:, k, :].ravel(order="C")
        flat = flat - flat.mean()
        z = np.cumsum(flat)
        walks_per_shell.append(z)
        diag = sublevel_persistence_1d(z)
        life = persistence_lifetimes(diag)
        if life.size > 0:
            pooled_life.append(life)
    if not pooled_life:
        return {"lifetimes": np.array([]), "walks": walks_per_shell,
                 "walk_length": 0, "walk_std": 0.0}
    L = np.concatenate(pooled_life)
    walk_length = walks_per_shell[0].size
    walk_std = float(np.std(np.concatenate(walks_per_shell)))
    return {"lifetimes": L, "walks": walks_per_shell,
             "walk_length": walk_length, "walk_std": walk_std}


# ---------------------------------------------------------------------------
# Microstructure family generators
# ---------------------------------------------------------------------------

def simulate_voxel_displacements(
    kind: str, n_molecules: int, rng: np.random.Generator, **params,
) -> np.ndarray:
    """Sample 3D displacement vectors for one voxel with given micro."""
    if kind == "free":
        return displacements_free(
            params["D"], params["Delta"], n_molecules, rng,
        )
    if kind == "restricted":
        return displacements_restricted(
            params["D_perp"], params["D_par"], params["Delta"],
            n_molecules,
            np.asarray(params.get("fibre_dir", [0, 0, 1]), dtype=float),
            rng,
        )
    if kind == "stable":
        return displacements_stable(
            params["alpha"], params["D_alpha"], params["Delta"],
            n_molecules, rng,
        )
    if kind == "mixed":
        # convex combination of two compartments
        f1 = params["fraction_1"]
        n1 = int(round(f1 * n_molecules))
        n2 = n_molecules - n1
        sub1 = dict(params["params_1"])
        sub2 = dict(params["params_2"])
        sub1.setdefault("Delta", params.get("Delta"))
        sub2.setdefault("Delta", params.get("Delta"))
        d1 = simulate_voxel_displacements(
            params["kind_1"], n1, rng, **sub1,
        )
        d2 = simulate_voxel_displacements(
            params["kind_2"], n2, rng, **sub2,
        )
        return np.concatenate([d1, d2], axis=0)
    raise ValueError(kind)


def simulate_roi(
    kind: str, n_voxels: int, n_dirs: int, n_molecules: int,
    b_values: np.ndarray, Delta: float, snr: float,
    param_jitter: float, rng: np.random.Generator, **params,
) -> Dict[str, np.ndarray]:
    """Simulate one ROI (population of voxels with a common microstructure
    plus log-normal parameter jitter).

    Output: (n_voxels, n_shells, n_dirs) HARDI signal with Rician noise.
    """
    directions = fibonacci_sphere(n_dirs)
    S_clean = np.zeros((n_voxels, b_values.size, n_dirs))
    for v in range(n_voxels):
        local = {}
        for k, vv in params.items():
            if isinstance(vv, (int, float, np.integer, np.floating)):
                local[k] = float(vv) * float(np.exp(
                    rng.normal(0.0, param_jitter)
                ))
                if k == "alpha":
                    local[k] = float(np.clip(local[k], 0.3, 2.0))
            else:
                local[k] = vv
        disp = simulate_voxel_displacements(
            kind, n_molecules, rng, Delta=Delta, **local,
        )
        S_clean[v] = hardi_signal(disp, b_values, directions, Delta=Delta)
    # Rician noise at SNR (at b=0)
    sigma = 1.0 / snr
    nr = rng.normal(0.0, sigma, size=S_clean.shape)
    ni = rng.normal(0.0, sigma, size=S_clean.shape)
    S_noisy = np.sqrt((S_clean + nr) ** 2 + ni ** 2)
    return {"S": S_noisy, "S_clean": S_clean, "directions": directions,
             "b_values": b_values}


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def hill_alpha(life: np.ndarray, k_frac: float = 0.10,
                min_k: int = 20) -> float:
    L = np.sort(life)[::-1]
    if L.size < min_k + 1:
        return np.nan
    k = max(min_k, int(k_frac * L.size))
    k = min(k, L.size - 1)
    m = float(np.mean(np.log(L[:k]) - np.log(L[k])))
    return 1.0 / m if m > 0 else np.nan


def evaluate_roi(S: np.ndarray, b_values: np.ndarray) -> Dict[str, float]:
    out = build_roi_walk_and_lifetimes(S, b_values)
    L = out["lifetimes"]
    if L.size < 25:
        return {"alpha_hill": np.nan, "xi": np.nan, "W1": np.nan,
                 "n_lifetimes": int(L.size)}
    return {
        "alpha_hill": hill_alpha(L),
        "xi": GenParetoShapeEstimator(k_fraction=0.20, min_k=15)
              .fit(L)["xi"],
        "W1": WassersteinAnomalyEstimator().fit(
            L, path_length=out["walk_length"],
            path_std=out["walk_std"],
        )["W1"],
        "n_lifetimes": int(L.size),
    }


# ---------------------------------------------------------------------------
# Microstructure roster
# ---------------------------------------------------------------------------

def microstructure_roster():
    """Roster of microstructures to test. Each entry is
    (label, kind, params)."""
    return [
        # Free Gaussian diffusivity sweep (cortex-like to ventricle-like)
        ("free_D_0.7", "free", dict(D=0.7e-3)),
        ("free_D_1.0", "free", dict(D=1.0e-3)),
        ("free_D_1.5", "free", dict(D=1.5e-3)),
        # Restricted (axon-like): low D_perp, high D_par
        ("restricted_axon_R_strong", "restricted",
         dict(D_perp=0.10e-3, D_par=1.7e-3)),
        ("restricted_axon_R_medium", "restricted",
         dict(D_perp=0.20e-3, D_par=1.7e-3)),
        ("restricted_axon_R_weak", "restricted",
         dict(D_perp=0.40e-3, D_par=1.7e-3)),
        # Alpha-stable (anomalous diffusion) baseline
        ("stable_alpha_1.5", "stable",
         dict(alpha=1.5, D_alpha=0.8e-3)),
        ("stable_alpha_1.8", "stable",
         dict(alpha=1.8, D_alpha=0.8e-3)),
        # Mixed compartments: 70% restricted axon + 30% free
        ("mixed_70axon_30free", "mixed", dict(
            fraction_1=0.7,
            kind_1="restricted",
            params_1=dict(D_perp=0.15e-3, D_par=1.7e-3),
            kind_2="free", params_2=dict(D=0.8e-3),
        )),
    ]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    n_voxels = 60
    n_dirs = 90
    n_molecules = 4000
    Delta = 0.05
    snr = 30.0
    b_values = np.asarray(HCP_SHELLS, dtype=float)  # (0, 1000, 2000, 3000)
    n_reps = 10  # 10 ROIs per microstructure (proxy for subjects)

    roster = microstructure_roster()
    rows = []
    for label, kind, params in roster:
        print(f"--- {label} ({kind}) ---", flush=True)
        for r in range(n_reps):
            rng = np.random.default_rng(hash((label, r)) % (2 ** 31))
            roi = simulate_roi(
                kind, n_voxels=n_voxels, n_dirs=n_dirs,
                n_molecules=n_molecules, b_values=b_values,
                Delta=Delta, snr=snr, param_jitter=0.05, rng=rng,
                **params,
            )
            res = evaluate_roi(roi["S"], b_values)
            res.update({"family": label, "kind": kind, "rep": r})
            rows.append(res)

    out_path = ROOT / "results" / "c1_biophysical_testbed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out_path} ({len(rows)} rows)")

    # Summary
    import collections
    by_fam = collections.defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    print(f"\n{'family':30s} {'alpha_hill':>14s} {'xi':>14s} {'W1':>14s}")
    print("-" * 75)
    for fam in [lab for (lab, _, _) in roster]:
        rs = by_fam[fam]
        a = np.array([r["alpha_hill"] for r in rs])
        x = np.array([r["xi"] for r in rs])
        w = np.array([r["W1"] for r in rs])
        print(
            f"{fam:30s} "
            f"{np.nanmean(a):>6.3f}+-{np.nanstd(a):>5.3f}  "
            f"{np.nanmean(x):>6.3f}+-{np.nanstd(x):>5.3f}  "
            f"{np.nanmean(w):>6.3f}+-{np.nanstd(w):>5.3f}"
        )


if __name__ == "__main__":
    main()
