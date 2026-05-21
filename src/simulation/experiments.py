"""Simulation experiments for the DW-MRI persistent-homology paper.

Each ``run_experiment_X`` function returns a dict that is serialised to
``results/experiment_X.npz`` and later consumed by ``analysis/make_figures.py``.
The experiments are deliberately small enough to run in minutes on a
laptop; full-scale versions are produced on Sherlock SLURM (see
``sherlock/``).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Tuple

from ..generators.dwmri_signal import (
    DEFAULT_BVALUES,
    gaussian_signal,
    kurtosis_signal,
    stretched_exponential_signal,
    stable_displacement_signal,
    sample_stable_displacement,
    signal_from_displacements,
)
from ..generators.hardi import (
    HCP_SHELLS,
    fibonacci_sphere,
    displacements_free,
    displacements_stable,
    displacements_restricted,
    hardi_signal,
)
from ..simulation.region_simulator import simulate_region
from ..estimators.dwmri_estimators import (
    PersistenceTailDWMRI,
    KurtosisFit,
    StretchedExponentialFit,
)
from ..utils.parent_bridge import (
    sublevel_persistence_1d,
    persistence_lifetimes,
    stable_levy_process,
)


# ===========================================================================
# Experiment 1: parent-paper replication on stable Levy paths
# ===========================================================================

def run_experiment_1(
    alphas: Tuple[float, ...] = (1.2, 1.5, 1.8, 2.0),
    n_replications: int = 50,
    n_steps: int = 2000,
    seed: int = 20260420,
) -> Dict:
    """Replicate the parent paper: persistence-tail on stable Levy paths.

    For each alpha, simulate ``n_replications`` symmetric alpha-stable
    paths of length ``n_steps``, compute sublevel persistence per path,
    pool lifetimes across replications, and report Hill-estimator
    alpha_hat with bootstrap CI.
    """
    rng = np.random.default_rng(seed)
    out_alphas = []
    out_alpha_hats = []
    out_alpha_lows = []
    out_alpha_highs = []
    out_n_lifetimes = []
    for alpha in alphas:
        pooled = []
        for _ in range(n_replications):
            _, path = stable_levy_process(
                alpha=alpha, n_steps=n_steps, dt=1.0 / n_steps, d=1, rng=rng,
            )
            diag = sublevel_persistence_1d(path[:, 0])
            life = persistence_lifetimes(diag)
            if life.size > 0:
                pooled.append(life)
        L = np.concatenate(pooled) if pooled else np.array([])
        L.sort()
        L = L[::-1]
        N = L.size
        k = max(20, int(0.1 * N))
        k = min(k, N - 1)
        if N < 20:
            ah = ahi = ahl = float("nan")
        else:
            m = float(np.mean(np.log(L[:k]) - np.log(L[k])))
            if m <= 0:
                ah = float("nan")
                ahl = ahi = float("nan")
            else:
                ah = 1.0 / m
                se = ah / np.sqrt(k)
                ahl = ah - 1.96 * se
                ahi = ah + 1.96 * se
        out_alphas.append(alpha)
        out_alpha_hats.append(ah)
        out_alpha_lows.append(ahl)
        out_alpha_highs.append(ahi)
        out_n_lifetimes.append(N)

    return {
        "name": "experiment_1_parent_replication",
        "alphas": np.asarray(out_alphas),
        "alpha_hats": np.asarray(out_alpha_hats),
        "alpha_ci_low": np.asarray(out_alpha_lows),
        "alpha_ci_high": np.asarray(out_alpha_highs),
        "n_lifetimes": np.asarray(out_n_lifetimes),
        "n_replications": n_replications,
        "n_steps": n_steps,
    }


# ===========================================================================
# Experiment 2: signal-model discrimination on dense S(b) curves
# ===========================================================================

def run_experiment_2(
    bmax: float = 3000.0,
    n_b: int = 7,
    D: float = 0.8e-3,
    K: float = 1.0,
    alpha_stable: Tuple[float, ...] = (1.2, 1.5, 1.8),
    alpha_se: float = 1.5,
    snr: float = 30.0,
    n_trials: int = 200,
    seed: int = 20260421,
) -> Dict:
    """Compare DKI K and stretched-alpha estimation across signal models.

    For each model (Gaussian, DKI K=K, stretched alpha_se, stable
    alpha=1.2, 1.5, 1.8) we generate ``n_trials`` noisy realisations of
    a single-direction S(b) curve, fit K and alpha_se, and report the
    bias/variance of each estimator.
    """
    rng = np.random.default_rng(seed)
    b = np.array(DEFAULT_BVALUES[:n_b], dtype=float)
    models = []
    # Gaussian baseline
    models.append(("gaussian", dict(D=D)))
    models.append(("dki", dict(D=D, K=K)))
    models.append(("stretched", dict(D=D, alpha_se=alpha_se)))
    for a in alpha_stable:
        models.append((f"stable_a{a:.1f}", dict(D_alpha=D, alpha=a)))

    results: Dict[str, Dict[str, NDArray]] = {}
    for name, p in models:
        K_hats = []
        alpha_se_hats = []
        for _ in range(n_trials):
            if name == "gaussian":
                S = gaussian_signal(b, p["D"])
            elif name == "dki":
                S = kurtosis_signal(b, p["D"], p["K"])
            elif name == "stretched":
                S = stretched_exponential_signal(b, p["D"], p["alpha_se"])
            else:  # stable_*
                S = stable_displacement_signal(b, p["alpha"], p["D_alpha"],
                                                Delta=0.05)
            sigma = 1.0 / snr
            nr = rng.normal(0.0, sigma, size=S.shape)
            ni = rng.normal(0.0, sigma, size=S.shape)
            S_noisy = np.sqrt((S + nr) ** 2 + ni ** 2)
            K_hats.append(KurtosisFit().fit(b, S_noisy)["K"])
            alpha_se_hats.append(
                StretchedExponentialFit().fit(b, S_noisy)["alpha_se"]
            )
        results[name] = {
            "K_hat": np.asarray(K_hats),
            "alpha_se_hat": np.asarray(alpha_se_hats),
            "params": p,
        }
    return {
        "name": "experiment_2_signal_models",
        "b_values": b,
        "results": results,
        "snr": snr,
        "n_trials": n_trials,
    }


# ===========================================================================
# Experiment 3: region pooling of persistence lifetimes for microstructure
# ===========================================================================

def run_experiment_3(
    kinds: Tuple[Tuple[str, Dict], ...] = (
        ("free",       dict(D=0.8e-3)),
        ("restricted", dict(D_perp=0.2e-3, D_par=1.7e-3, fibre_dir=[0, 0, 1])),
        ("stable",     dict(alpha=1.8, D_alpha=0.8e-3)),
        ("stable",     dict(alpha=1.5, D_alpha=0.8e-3)),
        ("stable",     dict(alpha=1.2, D_alpha=0.8e-3)),
    ),
    n_voxels: int = 60,
    n_dirs: int = 90,
    snr: float = 30.0,
    seed: int = 20260422,
) -> Dict:
    """For each microstructural model, simulate a region and compute three
    pooled non-Gaussianity diagnostics:

    * alpha_hat_cumulative  -- persistence-tail in cumulative mode
    * alpha_hat_directional -- persistence-tail in directional mode
    * mean kurtosis K_dir   -- mean of single-direction DKI K
    """
    rng = np.random.default_rng(seed)
    diagnostics = []
    for kind, p in kinds:
        res = simulate_region(
            kind, n_voxels=n_voxels, n_dirs=n_dirs, n_molecules=10000,
            snr=snr, rng=rng, param_jitter=0.05, **p,
        )
        S = res["S"]            # (n_voxels, n_shells, n_dirs)
        b_values = res["b_values"]
        # Cumulative-mode persistence
        cum = PersistenceTailDWMRI(mode="cumulative", k_fraction=0.10).fit(S)
        # Directional-mode persistence (per-shell, per-voxel)
        # shape (n_voxels * (n_shells - 1), n_dirs)
        Sd = S[:, 1:, :].reshape(-1, S.shape[2])
        dirmode = PersistenceTailDWMRI(
            mode="directional", k_fraction=0.10
        ).fit(Sd)
        # Mean kurtosis from per-direction DKI fits, averaged across voxels
        Ks = []
        alpha_se_ls = []
        for v in range(S.shape[0]):
            S_dir_mean = S[v].mean(axis=1)
            kf = KurtosisFit().fit(b_values, S_dir_mean)
            se = StretchedExponentialFit().fit(b_values, S_dir_mean)
            if np.isfinite(kf["K"]):
                Ks.append(kf["K"])
            if np.isfinite(se["alpha_se"]):
                alpha_se_ls.append(se["alpha_se"])
        diagnostics.append({
            "kind": kind,
            "params": {k_: float(v) if isinstance(v, (int, float)) else v
                       for k_, v in p.items()},
            "alpha_cumulative": float(cum["alpha_hat"]),
            "alpha_directional": float(dirmode["alpha_hat"]),
            "mean_K": float(np.mean(Ks)) if Ks else float("nan"),
            "std_K": float(np.std(Ks)) if Ks else float("nan"),
            "mean_alpha_se": (
                float(np.mean(alpha_se_ls)) if alpha_se_ls else float("nan")
            ),
            "std_alpha_se": (
                float(np.std(alpha_se_ls)) if alpha_se_ls else float("nan")
            ),
            "n_lifetimes_cum": int(cum["n_lifetimes"]),
            "n_lifetimes_dir": int(dirmode["n_lifetimes"]),
        })
    return {
        "name": "experiment_3_region_pooling",
        "diagnostics": diagnostics,
        "n_voxels": n_voxels,
        "n_dirs": n_dirs,
        "snr": snr,
    }


# ===========================================================================
# Experiment 4: calibration sweep alpha -> (K, alpha_se, persistence)
# ===========================================================================

def run_experiment_4(
    alpha_grid: Tuple[float, ...] = (1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 1.9, 2.0),
    D: float = 0.8e-3,
    n_voxels: int = 40,
    n_dirs: int = 90,
    snr: float = 30.0,
    seed: int = 20260423,
) -> Dict:
    """Sweep alpha and record (K_hat, alpha_se_hat, alpha_persistence_hat).

    Produces the calibration curve used to discuss the joint behaviour of
    classical DKI / stretched-exponential non-Gaussianity measures and the
    proposed persistence-tail exponent.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for a in alpha_grid:
        if a >= 2.0:
            kind = "free"
            params = dict(D=D)
        else:
            kind = "stable"
            params = dict(alpha=float(a), D_alpha=D)
        res = simulate_region(
            kind, n_voxels=n_voxels, n_dirs=n_dirs, n_molecules=10000,
            snr=snr, rng=rng, param_jitter=0.05, **params,
        )
        S = res["S"]
        b_values = res["b_values"]
        cum = PersistenceTailDWMRI(mode="cumulative", k_fraction=0.10).fit(S)
        Ks = []
        ses = []
        for v in range(S.shape[0]):
            S_mean = S[v].mean(axis=1)
            kf = KurtosisFit().fit(b_values, S_mean)
            sef = StretchedExponentialFit().fit(b_values, S_mean)
            if np.isfinite(kf["K"]):
                Ks.append(kf["K"])
            if np.isfinite(sef["alpha_se"]):
                ses.append(sef["alpha_se"])
        rows.append({
            "alpha_true": float(a),
            "alpha_persistence": float(cum["alpha_hat"]),
            "alpha_persistence_lo": float(cum["alpha_ci_low"]),
            "alpha_persistence_hi": float(cum["alpha_ci_high"]),
            "K_mean": float(np.mean(Ks)) if Ks else float("nan"),
            "K_std": float(np.std(Ks)) if Ks else float("nan"),
            "alpha_se_mean": float(np.mean(ses)) if ses else float("nan"),
            "alpha_se_std": float(np.std(ses)) if ses else float("nan"),
        })
    return {
        "name": "experiment_4_calibration",
        "rows": rows,
        "D": D,
        "n_voxels": n_voxels,
        "n_dirs": n_dirs,
        "snr": snr,
    }


# ===========================================================================
# Serialisation helpers
# ===========================================================================

def _to_python(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, dict):
        return {k: _to_python(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_python(x) for x in o]
    return o


def save_experiment(result: Dict, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = result["name"]
    p = out_dir / f"{name}.json"
    with open(p, "w") as f:
        json.dump(_to_python(result), f, indent=2)
    return p
