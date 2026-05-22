"""Estimators that work at the Gaussian boundary of the alpha-stable family.

The Hill estimator used elsewhere in this project is consistent for the
stability index alpha when alpha < 2 (heavy-tail regime). It is degenerate
at alpha = 2, where the underlying lifetime tail is exponential rather
than power-law (parent paper Remark on Gaussian boundary).

Two estimators here generalise it across the boundary:

* :class:`GenParetoShapeEstimator` -- fits a generalised Pareto distribution
  to the top-k persistence lifetimes and returns the shape parameter xi.
  In the heavy-tail (alpha < 2) regime, xi = 1/alpha > 0; at the Gaussian
  boundary, xi -> 0 (exponential tail); under bounded support (heavy
  restriction), xi < 0. The single scalar xi therefore varies smoothly
  across the regime that breaks the Hill estimator. Consistency theory
  for the peaks-over-threshold fit is in Smith (1987) and Drees, de Haan,
  Resnick (2000).

* :class:`WassersteinAnomalyEstimator` -- 1-Wasserstein distance between
  the empirical persistence lifetime distribution and a reference
  Brownian-bridge lifetime distribution computed from a long simulated
  reference. A pure Brownian bridge gives W_1 = 0 (up to finite-sample
  noise); any departure from the Brownian limit (restriction,
  heterogeneity, anomalous diffusion) gives W_1 > 0. The reference is
  cached on first use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.stats import genpareto, wasserstein_distance


# ---------------------------------------------------------------------------
# Generalised-Pareto shape estimator
# ---------------------------------------------------------------------------

@dataclass
class GenParetoShapeEstimator:
    """Peaks-over-threshold fit of the generalised Pareto distribution to
    the top fraction of persistence lifetimes.

    Returns the shape parameter ``xi`` and scale parameter ``sigma``:

    * xi > 0  -> heavy-tailed regime (alpha = 1/xi < infinity)
    * xi = 0  -> exponential (Gaussian boundary)
    * xi < 0  -> bounded support (restricted-diffusion-like)
    """

    k_fraction: float = 0.20
    min_k: int = 15

    def fit(self, lifetimes: NDArray[np.float64]) -> Dict[str, float]:
        life = np.asarray(lifetimes, dtype=float)
        life = life[np.isfinite(life) & (life > 0)]
        if life.size < self.min_k + 1:
            return {"xi": np.nan, "sigma": np.nan, "threshold": np.nan,
                    "n_lifetimes": life.size, "k_used": 0}
        life_sorted = np.sort(life)[::-1]  # descending
        N = life_sorted.size
        k = max(self.min_k, int(self.k_fraction * N))
        k = min(k, N - 1)
        threshold = life_sorted[k]
        exceedances = life_sorted[:k] - threshold
        # genpareto.fit returns (c, loc, scale); we fix loc=0 (exceedances)
        try:
            xi, _, sigma = genpareto.fit(exceedances, floc=0)
        except Exception:
            return {"xi": np.nan, "sigma": np.nan, "threshold": float(threshold),
                    "n_lifetimes": N, "k_used": k}
        return {
            "xi": float(xi),
            "sigma": float(sigma),
            "threshold": float(threshold),
            "n_lifetimes": int(N),
            "k_used": int(k),
        }


# ---------------------------------------------------------------------------
# Wasserstein-to-Brownian-reference estimator
# ---------------------------------------------------------------------------

class WassersteinAnomalyEstimator:
    """1-Wasserstein distance from the empirical persistence-lifetime
    distribution to a Brownian-bridge reference of the same length.

    The reference is generated once per (length, rng_seed) pair by
    simulating ``n_reference_paths`` Brownian bridges of length ``n``
    and pooling their persistence lifetimes. Lifetimes are rescaled by
    the path standard deviation before comparison, so the resulting
    distance is dimensionless and reflects shape rather than overall
    scale.
    """

    _reference_cache: Dict[int, NDArray[np.float64]] = {}

    def __init__(
        self,
        n_reference_paths: int = 50,
        rng_seed: int = 20260522,
        max_reference_length: int = 10_000,
        empirical_subsample: int = 20_000,
    ):
        """The Brownian-bridge reference paths are capped at
        ``max_reference_length`` to keep persistence computation tractable;
        the in-vivo empirical lifetimes are subsampled to
        ``empirical_subsample`` before the Wasserstein computation. Both
        are scale-equivariant: the ROI walk has been pre-scaled by its
        standard deviation in :meth:`fit`, and the cached reference is
        pre-scaled by its own standard deviation, so W1 is dimensionless.
        """
        self.n_reference_paths = n_reference_paths
        self.rng_seed = rng_seed
        self.max_reference_length = int(max_reference_length)
        self.empirical_subsample = int(empirical_subsample)

    def _get_reference(self, n: int) -> NDArray[np.float64]:
        """Return a long pooled vector of Brownian-bridge persistence
        lifetimes (rescaled to unit variance). The reference path length
        is capped at ``self.max_reference_length`` for tractability.
        """
        n_capped = min(int(n), self.max_reference_length)
        if n_capped in self._reference_cache:
            return self._reference_cache[n_capped]
        from src.utils.parent_bridge import (
            sublevel_persistence_1d,
            persistence_lifetimes,
        )
        rng = np.random.default_rng(self.rng_seed)
        all_life = []
        for _ in range(self.n_reference_paths):
            increments = rng.normal(0.0, 1.0, size=n_capped)
            increments -= increments.mean()
            path = np.cumsum(increments)
            sigma = path.std() or 1.0
            life = persistence_lifetimes(sublevel_persistence_1d(path)) / sigma
            if life.size > 0:
                all_life.append(life)
        ref = np.sort(np.concatenate(all_life)) if all_life else np.array([1.0])
        self._reference_cache[n_capped] = ref
        return ref

    def fit(
        self,
        lifetimes: NDArray[np.float64],
        path_length: Optional[int] = None,
        path_std: Optional[float] = None,
    ) -> Dict[str, float]:
        life = np.asarray(lifetimes, dtype=float)
        life = life[np.isfinite(life) & (life > 0)]
        if life.size < 20 or path_length is None:
            return {"W1": np.nan, "n_lifetimes": int(life.size)}
        sigma = path_std if (path_std is not None and path_std > 0) else 1.0
        empirical = life / sigma
        # Subsample if the empirical sample is huge: Wasserstein on >1e5
        # samples is unnecessarily expensive and the distance estimate is
        # already converged.
        if empirical.size > self.empirical_subsample:
            rng = np.random.default_rng(self.rng_seed + 1)
            idx = rng.choice(empirical.size, self.empirical_subsample,
                              replace=False)
            empirical = empirical[idx]
        empirical = np.sort(empirical)
        reference = self._get_reference(int(path_length))
        try:
            d = wasserstein_distance(empirical, reference)
        except Exception:
            return {"W1": np.nan, "n_lifetimes": int(life.size)}
        return {
            "W1": float(d),
            "n_lifetimes": int(life.size),
        }
