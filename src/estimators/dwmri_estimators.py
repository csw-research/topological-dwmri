"""Estimators applied to diffusion-weighted MRI signals.

We provide three estimators that play complementary roles:

* :class:`PersistenceTailDWMRI` -- the contribution of this paper. It
  estimates the tail exponent alpha_hat of the persistent-homology
  lifetime distribution of the directional DW-MRI signal, viewed as a
  function of unit gradient direction on the sphere at a fixed shell.

* :class:`KurtosisFit` -- standard diffusion-kurtosis (DKI) regression in
  the form ln S(b) = ln S0 - b D + (1/6) b^2 D^2 K. Used as a comparison
  benchmark and a smoothness baseline.

* :class:`StretchedExponentialFit` -- the stretched-exponential model
  S(b) = S0 exp(-(b D)^(alpha_se / 2)). The exponent alpha_se / 2 is the
  classical phenomenological non-Gaussianity index.

The PersistenceTailDWMRI estimator imports
``sublevel_persistence_1d`` and ``persistence_lifetimes`` from the parent
paper without modification.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Optional, Sequence

from scipy.optimize import curve_fit

from ..utils.parent_bridge import (
    sublevel_persistence_1d,
    persistence_lifetimes,
)


# ---------------------------------------------------------------------------
# Persistence-tail estimator
# ---------------------------------------------------------------------------

class PersistenceTailDWMRI:
    """Persistence-tail exponent alpha_hat for DW-MRI signals.

    Three analysis modes are supported:

    * ``mode='cumulative'`` (default, recommended): the input is a 3D
      array ``S[voxel, shell, direction]`` (the b=0 image is the first
      shell). For each (voxel, shell) we form the per-direction attenuation
      ``y_{k,i} = -ln(S_{k,i} / S0)`` and the *cumulative sum*
      ``Z_{k,j} = sum_{i <= j} (y_{k,i} - mean_i y_{k,i})``. Under
      isotropy of the underlying molecular-displacement distribution, the
      centred ``y_{k,i}`` are exchangeable across directions; if the
      displacement is symmetric alpha-stable the generalised central limit
      theorem makes ``Z_k`` converge to a Levy bridge whose lifetime tail
      has exponent alpha (Theorem 1 of the parent paper). We pool
      sublevel-persistence lifetimes across (voxels, shells) and apply the
      Hill estimator.

    * ``mode='directional'``: input shape ``(n_curves, n_dirs)``. Sublevel
      persistence is computed for each row of ``-ln(S/S0)``. Useful for
      anisotropic voxels and as a complement to FA.

    * ``mode='bvalue'``: input shape ``(n_curves, n_b)``. Sublevel
      persistence is computed for each voxel's b-curve.
    """

    _VALID_MODES = {"cumulative", "directional", "bvalue"}

    def __init__(
        self,
        mode: str = "cumulative",
        k_fraction: float = 0.15,
        min_k: int = 10,
        eps: float = 1e-8,
    ):
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"mode must be one of {self._VALID_MODES}, got {mode}"
            )
        self.mode = mode
        self.k_fraction = k_fraction
        self.min_k = min_k
        self.eps = eps
        self.alpha_hat: Optional[float] = None
        self.lifetimes_: Optional[NDArray[np.float64]] = None
        self.k_used: Optional[int] = None

    # -- core computation --------------------------------------------------

    @staticmethod
    def _signal_to_curve(S: NDArray[np.float64], eps: float) -> NDArray[np.float64]:
        """Convert a signal segment to the attenuation curve.

        ``y = -ln(S / S0)`` with ``S0 = max(S)``.
        """
        S0 = float(np.max(S))
        if S0 <= 0:
            return np.zeros_like(S)
        ratio = np.clip(S / S0, eps, None)
        return -np.log(ratio)

    def _collect_lifetimes_curves(
        self, curves: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        all_lifetimes = []
        for row in curves:
            if row.size < 4:
                continue
            diagram = sublevel_persistence_1d(np.asarray(row, dtype=float))
            life = persistence_lifetimes(diagram)
            if life.size > 0:
                all_lifetimes.append(life)
        if not all_lifetimes:
            return np.array([])
        return np.concatenate(all_lifetimes)

    def _collect_lifetimes_cumulative(
        self, S: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Cumulative-sum analysis. ``S`` may have shape
        ``(voxel, shell, dir)`` or ``(shell, dir)``."""
        if S.ndim == 2:
            S = S[None, :, :]
        n_v, n_s, n_d = S.shape
        # use the first shell as the b=0 normalisation
        S0 = np.maximum(S[:, 0:1, :], self.eps)
        ratio = np.clip(S / S0, self.eps, None)
        y = -np.log(ratio)  # (v, s, d)
        # for each (v, s), drop the b=0 shell, centre across directions,
        # then cumulative sum to form Z_{k,j}.
        all_curves = []
        for v in range(n_v):
            for k in range(1, n_s):
                row = y[v, k, :]
                row = row - row.mean()
                Z = np.cumsum(row)
                all_curves.append(Z)
        return self._collect_lifetimes_curves(np.asarray(all_curves))

    # -- public API --------------------------------------------------------

    def fit(self, S: NDArray[np.float64]) -> Dict[str, float]:
        """Estimate alpha from a DW-MRI signal array.

        Accepted shapes depend on ``mode``:
        * cumulative : ``(voxel, shell, dir)`` or ``(shell, dir)``.
        * directional: ``(n_curves, n_dirs)`` or ``(n_dirs,)``.
        * bvalue     : ``(n_curves, n_b)`` or ``(n_b,)``.
        """
        S = np.asarray(S, dtype=float)
        if self.mode == "cumulative":
            life = self._collect_lifetimes_cumulative(S)
        else:
            if S.ndim == 1:
                S = S[None, :]
            curves = np.stack([self._signal_to_curve(r, self.eps) for r in S])
            life = self._collect_lifetimes_curves(curves)

        life = np.sort(life)[::-1]
        self.lifetimes_ = life
        N = life.size
        if N < self.min_k + 1:
            self.alpha_hat = np.nan
            return {
                "alpha_hat": np.nan, "alpha_ci_low": np.nan,
                "alpha_ci_high": np.nan, "n_lifetimes": N, "k_used": 0,
            }

        k = max(self.min_k, int(self.k_fraction * N))
        k = min(k, N - 1)
        self.k_used = k
        log_top = np.log(life[:k])
        log_thresh = np.log(life[k])
        hill_mean = float(np.mean(log_top - log_thresh))
        if hill_mean <= 0:
            self.alpha_hat = np.nan
            return {
                "alpha_hat": np.nan, "alpha_ci_low": np.nan,
                "alpha_ci_high": np.nan, "n_lifetimes": N, "k_used": k,
            }
        self.alpha_hat = 1.0 / hill_mean
        se = self.alpha_hat / np.sqrt(k)
        return {
            "alpha_hat": self.alpha_hat,
            "alpha_ci_low": self.alpha_hat - 1.96 * se,
            "alpha_ci_high": self.alpha_hat + 1.96 * se,
            "n_lifetimes": N,
            "k_used": k,
        }


# ---------------------------------------------------------------------------
# Diffusion-kurtosis fit
# ---------------------------------------------------------------------------

class KurtosisFit:
    """Single-direction DKI regression.

    Fits ln S(b) = ln S0 - b D + (1/6) b^2 D^2 K by ordinary least squares
    in the log domain. Returns D, K, and S0.
    """

    def __init__(self, b_max: float = 3000.0):
        self.b_max = b_max

    def fit(
        self,
        b: NDArray[np.float64],
        S: NDArray[np.float64],
    ) -> Dict[str, float]:
        b = np.asarray(b, dtype=float)
        S = np.asarray(S, dtype=float)
        m = (b <= self.b_max) & (S > 0)
        b = b[m]
        S = S[m]
        if b.size < 3:
            return {"D": np.nan, "K": np.nan, "S0": np.nan, "rss": np.nan}
        # Linear in (D, D^2 K / 6, ln S0):
        #   ln S = ln S0 - b D + b^2 (D^2 K / 6)
        # We use a non-linear fit because (D, K) are the natural quantities.
        try:
            def model(bv, S0, D, K):
                return S0 * np.exp(-bv * D + (1.0 / 6.0) * (bv * D) ** 2 * K)
            p0 = [float(S[np.argmin(b)]), 1e-3, 1.0]
            popt, _ = curve_fit(
                model, b, S, p0=p0,
                bounds=([0.0, 1e-6, -2.0], [np.inf, 5e-3, 6.0]),
                maxfev=2000,
            )
            S0_hat, D_hat, K_hat = popt
            resid = S - model(b, *popt)
            return {
                "S0": float(S0_hat),
                "D": float(D_hat),
                "K": float(K_hat),
                "rss": float(np.sum(resid ** 2)),
            }
        except Exception:
            return {"D": np.nan, "K": np.nan, "S0": np.nan, "rss": np.nan}


# ---------------------------------------------------------------------------
# Stretched-exponential fit
# ---------------------------------------------------------------------------

class StretchedExponentialFit:
    """Stretched-exponential regression S(b) = S0 exp(-(b D)^(alpha_se / 2))."""

    def fit(
        self,
        b: NDArray[np.float64],
        S: NDArray[np.float64],
    ) -> Dict[str, float]:
        b = np.asarray(b, dtype=float)
        S = np.asarray(S, dtype=float)
        m = (b >= 0) & (S > 0)
        b = b[m]
        S = S[m]
        if b.size < 3:
            return {"S0": np.nan, "D": np.nan, "alpha_se": np.nan,
                    "rss": np.nan}
        try:
            def model(bv, S0, D, alpha_se):
                return S0 * np.exp(-((bv * D) ** (alpha_se / 2.0)))
            p0 = [float(S[np.argmin(b)]), 1e-3, 1.5]
            popt, _ = curve_fit(
                model, b, S, p0=p0,
                bounds=([0.0, 1e-6, 0.3], [np.inf, 5e-3, 2.0]),
                maxfev=2000,
            )
            S0_hat, D_hat, alpha_hat = popt
            resid = S - model(b, *popt)
            return {
                "S0": float(S0_hat),
                "D": float(D_hat),
                "alpha_se": float(alpha_hat),
                "rss": float(np.sum(resid ** 2)),
            }
        except Exception:
            return {"S0": np.nan, "D": np.nan, "alpha_se": np.nan,
                    "rss": np.nan}


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def fit_all_estimators(
    b: NDArray[np.float64],
    S: NDArray[np.float64],
    directional_signal: Optional[NDArray[np.float64]] = None,
) -> Dict[str, Dict[str, float]]:
    """Run the three estimators on a single voxel.

    Parameters
    ----------
    b : array (n_b,)
        b-values for the single-direction signal ``S``.
    S : array (n_b,)
        Single-direction signal S(b).
    directional_signal : array (n_shells, n_dirs), optional
        If supplied, the persistence-tail estimator runs in the
        ``'directional'`` mode and uses this richer signal; otherwise it
        operates on the single-direction b-curve.
    """
    out: Dict[str, Dict[str, float]] = {}
    out["kurtosis"] = KurtosisFit().fit(b, S)
    out["stretched"] = StretchedExponentialFit().fit(b, S)
    if directional_signal is not None:
        out["persistence_tail"] = PersistenceTailDWMRI(
            mode="directional"
        ).fit(directional_signal)
    else:
        out["persistence_tail"] = PersistenceTailDWMRI(
            mode="bvalue"
        ).fit(S[None, :])
    return out
