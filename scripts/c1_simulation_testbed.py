#!/usr/bin/env python3
"""C1 simulation testbed: benchmark boundary-aware estimators against the
Hill estimator across the heavy-tail/Gaussian-boundary divide.

Workflow:

1. Simulate four bridge families of length n=2000, with 50 replicates each:
     (a) symmetric alpha-stable Levy bridges for alpha in {1.2, 1.5, 1.8, 2.0}
     (b) Brownian bridges with iid Gaussian increments (alpha = 2 ground
         truth, no restriction)
     (c) Brownian bridges with bounded-support increments (uniform on a
         finite interval) -- mimics restricted diffusion at the Gaussian
         boundary. Varying the interval width gives 'restricted radius'
         R in {0.5, 1.0, 2.0, 4.0}.
     (d) Mixed bridges: convex combination of restricted (60%) and free
         (40%) increments. Mimics a partial-volume voxel.

2. For each replicate, compute sublevel-set persistence and extract
   lifetimes. Then evaluate three estimators:
     * Hill (alpha_persistence) -- the parent paper's estimator
     * GenParetoShape (xi) -- new C1 primary
     * WassersteinAnomaly (W1) -- new C1 backup

3. Report mean +/- SD for each estimator on each family, and the
   reproducibility (Pearson r) of each estimator when the same
   underlying parameter is re-simulated.

The honest go/no-go for C1: does the new estimator separate restricted
from free Brownian (regime c vs regime b) with effect size that
matches or beats what the Hill estimator does in the alpha < 2 regime?
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
    stable_levy_process,
)
from src.estimators.boundary_estimators import (
    GenParetoShapeEstimator,
    WassersteinAnomalyEstimator,
)


# ---------------------------------------------------------------------------
# Bridge simulators
# ---------------------------------------------------------------------------

def levy_bridge(alpha: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Symmetric alpha-stable Levy bridge of length n."""
    _, path = stable_levy_process(
        alpha=alpha, n_steps=n, dt=1.0 / n, d=1, rng=rng,
    )
    p = path[:, 0]
    p = p - np.linspace(0, p[-1], len(p))  # bridge: pin endpoint to 0
    return p


def brownian_bridge(n: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Standard Brownian bridge with increment std sigma."""
    inc = rng.normal(0.0, sigma, size=n)
    inc -= inc.mean()
    return np.cumsum(inc)


def restricted_bridge(
    n: int, width: float, rng: np.random.Generator
) -> np.ndarray:
    """Bridge with bounded-support (uniform) increments. ``width`` is the
    half-width of the uniform distribution. As width -> 0 the path
    flattens; as width -> infinity it approaches a Brownian bridge (by
    the central limit theorem applied to the increments)."""
    inc = rng.uniform(-width, width, size=n)
    inc -= inc.mean()
    return np.cumsum(inc)


def mixed_bridge(
    n: int, restricted_frac: float, restricted_width: float,
    free_sigma: float, rng: np.random.Generator,
) -> np.ndarray:
    """Convex combination of restricted and free Gaussian increments."""
    n_r = int(restricted_frac * n)
    n_f = n - n_r
    inc_r = rng.uniform(-restricted_width, restricted_width, size=n_r)
    inc_f = rng.normal(0.0, free_sigma, size=n_f)
    inc = np.concatenate([inc_r, inc_f])
    rng.shuffle(inc)
    inc -= inc.mean()
    return np.cumsum(inc)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def hill_alpha(lifetimes: np.ndarray, k_frac: float = 0.10,
                min_k: int = 20) -> float:
    """Original Hill estimator on persistence lifetimes."""
    L = np.sort(lifetimes)[::-1]
    if L.size < min_k + 1:
        return np.nan
    k = max(min_k, int(k_frac * L.size))
    k = min(k, L.size - 1)
    m = float(np.mean(np.log(L[:k]) - np.log(L[k])))
    return 1.0 / m if m > 0 else np.nan


def evaluate_path(path: np.ndarray) -> Dict[str, float]:
    """All three estimators on one bridge realisation."""
    diag = sublevel_persistence_1d(path)
    life = persistence_lifetimes(diag)
    if life.size < 25:
        return {"alpha_hill": np.nan, "xi": np.nan, "W1": np.nan,
                 "n_lifetimes": int(life.size)}
    alpha = hill_alpha(life)
    gp = GenParetoShapeEstimator(k_fraction=0.20, min_k=15).fit(life)
    wa = WassersteinAnomalyEstimator().fit(
        life, path_length=path.size, path_std=path.std() or 1.0,
    )
    return {
        "alpha_hill": float(alpha),
        "xi": float(gp["xi"]),
        "W1": float(wa["W1"]),
        "n_lifetimes": int(life.size),
    }


# ---------------------------------------------------------------------------
# Simulation matrix
# ---------------------------------------------------------------------------

def run_family(name: str, simulator, n_replicates: int, rng_seed: int) -> List[Dict]:
    """Run one bridge family and return per-replicate results."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for r in range(n_replicates):
        path = simulator(rng)
        res = evaluate_path(path)
        res["family"] = name
        res["rep"] = r
        rows.append(res)
    return rows


def main():
    n = 2000
    n_reps = 50
    out = []

    for alpha in (1.2, 1.5, 1.8, 2.0):
        sim = lambda r, a=alpha: levy_bridge(a, n, r)
        out += run_family("levy_alpha_{:.1f}".format(alpha), sim,
                           n_reps, rng_seed=hash(("levy", alpha)) % (2 ** 31))

    out += run_family(
        "brownian_sigma_1.0",
        lambda r: brownian_bridge(n, sigma=1.0, rng=r),
        n_reps, rng_seed=2026_0522,
    )

    for w in (0.5, 1.0, 2.0, 4.0):
        sim = lambda r, ww=w: restricted_bridge(n, ww, r)
        out += run_family("restricted_w_{:.1f}".format(w), sim,
                           n_reps, rng_seed=hash(("restricted", w)) % (2 ** 31))

    out += run_family(
        "mixed_60r_w1.0_f0.5",
        lambda r: mixed_bridge(n, restricted_frac=0.6, restricted_width=1.0,
                                free_sigma=0.5, rng=r),
        n_reps, rng_seed=20260523,
    )

    out_path = ROOT / "results" / "c1_simulation_testbed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path} ({len(out)} rows)")

    # Quick summary
    import collections
    by_fam = collections.defaultdict(list)
    for r in out:
        by_fam[r["family"]].append(r)

    print(f"\n{'family':30s} {'alpha_hill':>14s} {'xi':>14s} {'W1':>14s}")
    print("-" * 75)
    for fam in sorted(by_fam.keys()):
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
