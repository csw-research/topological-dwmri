"""Minimal smoke tests for the DW-MRI pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generators.dwmri_signal import (
    gaussian_signal, stable_displacement_signal,
    sample_stable_displacement, signal_from_displacements,
    DEFAULT_BVALUES,
)
from src.estimators.dwmri_estimators import (
    PersistenceTailDWMRI, KurtosisFit, StretchedExponentialFit,
)
from src.simulation.region_simulator import simulate_region
from src.utils.parent_bridge import stable_levy_process


def test_gaussian_signal_round_trip():
    b = np.array(DEFAULT_BVALUES)
    S = gaussian_signal(b, D=0.7e-3)
    assert np.isclose(S[0], 1.0)
    assert np.all(np.diff(S) <= 0)


def test_stable_signal_matches_empirical():
    rng = np.random.default_rng(0)
    b = np.array(DEFAULT_BVALUES)
    for alpha in [1.2, 1.5, 1.8, 2.0]:
        disp = sample_stable_displacement(alpha, 0.7e-3, 0.05, 50000, rng)
        S_emp = signal_from_displacements(b, disp, Delta=0.05)
        S_an = stable_displacement_signal(b, alpha, 0.7e-3, Delta=0.05)
        assert np.max(np.abs(S_emp - S_an)) < 0.01


def test_kurtosis_fit_perfect_signal():
    b = np.array(DEFAULT_BVALUES)
    from src.generators.dwmri_signal import kurtosis_signal
    S = kurtosis_signal(b, D=0.7e-3, K=1.0)
    out = KurtosisFit().fit(b, S)
    assert abs(out["K"] - 1.0) < 1e-3
    assert abs(out["D"] - 0.7e-3) < 1e-6


def test_persistence_tail_on_stable_paths():
    rng = np.random.default_rng(1)
    from src.utils.parent_bridge import (
        sublevel_persistence_1d, persistence_lifetimes,
    )
    pooled = []
    for _ in range(20):
        _, path = stable_levy_process(
            alpha=1.5, n_steps=1000, dt=1.0 / 1000, d=1, rng=rng,
        )
        life = persistence_lifetimes(sublevel_persistence_1d(path[:, 0]))
        pooled.append(life)
    L = np.sort(np.concatenate(pooled))[::-1]
    k = int(0.1 * L.size)
    m = np.mean(np.log(L[:k]) - np.log(L[k]))
    alpha_hat = 1.0 / m
    assert 1.2 < alpha_hat < 1.7


def test_simulate_region_shapes():
    rng = np.random.default_rng(2)
    res = simulate_region(
        "free", n_voxels=5, n_dirs=20, n_molecules=1000, snr=30.0,
        rng=rng, D=0.7e-3,
    )
    assert res["S"].shape == (5, 4, 20)
    assert res["S_clean"].shape == (5, 4, 20)


if __name__ == "__main__":
    import sys as _sys
    for name in [
        "test_gaussian_signal_round_trip",
        "test_stable_signal_matches_empirical",
        "test_kurtosis_fit_perfect_signal",
        "test_persistence_tail_on_stable_paths",
        "test_simulate_region_shapes",
    ]:
        try:
            globals()[name]()
            print(f"PASS {name}")
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            _sys.exit(1)
