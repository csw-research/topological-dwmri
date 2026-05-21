"""Forward models for the diffusion-weighted MRI signal S(b).

We parameterise the diffusion-weighted signal in three complementary ways:

1. **Compartment forward models** that produce S(b) directly from a physical
   model of water displacement: free Gaussian, diffusion-kurtosis,
   stretched-exponential, restricted cylinders, restricted spheres,
   alpha-stable displacement, and mixtures.

2. **Displacement samplers** that generate molecular displacement
   trajectories from each model. The DW-MRI signal is then computed from
   the empirical characteristic function of the displacement,
       S(b) = E[exp(i q . x)],   q = sqrt(b / Delta) g_hat,
   so that the connection to the parent paper (where x is alpha-stable)
   is explicit.

Units throughout are physically meaningful: b in s/mm^2, D in mm^2/s,
Delta (diffusion time) in s. We adopt the narrow-pulse approximation:
S(b) is the Fourier transform of the displacement propagator.

References
----------
- Stejskal & Tanner (1965), J. Chem. Phys. 42, 288.
- Jensen, Helpern, Ramani, Lu, Kaczynski (2005), MRM 53, 1432.
- Bennett et al. (2003), MRM 50, 727 (stretched exponential).
- Soderman & Jonsson (1995), J. Magn. Reson. A 117, 94 (cylinders).
- Murday & Cotts (1968), J. Chem. Phys. 48, 4938 (spheres).
- Magin et al. (2008), Magn. Reson. Imaging 26, 1431 (anomalous diffusion).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Sequence, Tuple

# Re-export the parent paper's stable sampler.
from ..utils.parent_bridge import stable_rvs


# ---------------------------------------------------------------------------
# 1. Closed-form attenuation models in b-space
# ---------------------------------------------------------------------------

def gaussian_signal(
    b: NDArray[np.float64],
    D: float,
    S0: float = 1.0,
) -> NDArray[np.float64]:
    """Stejskal-Tanner mono-exponential decay.

    S(b) = S0 * exp(-b D).
    """
    b = np.asarray(b, dtype=float)
    return S0 * np.exp(-b * D)


def kurtosis_signal(
    b: NDArray[np.float64],
    D: float,
    K: float,
    S0: float = 1.0,
) -> NDArray[np.float64]:
    """Diffusion-kurtosis imaging signal (Jensen et al., 2005).

    ln S(b) = ln S0 - b D + (1/6) b^2 D^2 K.
    """
    b = np.asarray(b, dtype=float)
    return S0 * np.exp(-b * D + (1.0 / 6.0) * (b * D) ** 2 * K)


def stretched_exponential_signal(
    b: NDArray[np.float64],
    D: float,
    alpha_se: float,
    S0: float = 1.0,
) -> NDArray[np.float64]:
    """Stretched-exponential model (Bennett et al., 2003).

    S(b) = S0 * exp(-(b D)^(alpha_se / 2)).

    Note that ``alpha_se`` here is the *stretching exponent* with range
    (0, 2]; the value 2 recovers Gaussian diffusion.
    """
    b = np.asarray(b, dtype=float)
    return S0 * np.exp(-((b * D) ** (alpha_se / 2.0)))


# ---------------------------------------------------------------------------
# 2. Restricted diffusion in idealised geometries
# ---------------------------------------------------------------------------

def restricted_cylinder_signal(
    b: NDArray[np.float64],
    Delta: float,
    radius: float,
    D_intra: float,
    S0: float = 1.0,
    n_terms: int = 30,
) -> NDArray[np.float64]:
    """Signal from water restricted in an impermeable cylinder, gradient
    perpendicular to the cylinder axis. Soderman--Jonsson long-Delta limit.

    S(q) = sum_k a_k exp(-beta_k^2 D Delta / R^2 + ...)

    A good practical approximation in the long-pulse limit is
        S(b) ~ exp(-b D_perp_eff(Delta, R))
    where the effective perpendicular diffusivity tends to R^2 / Delta
    as Delta -> infinity. Here we implement a low-order Gaussian-phase
    approximation (Stepisnik) that is accurate for the b-values used in
    typical multi-shell HCP/MGH protocols.

    Parameters
    ----------
    b : array_like
        b-values in s/mm^2.
    Delta : float
        Effective diffusion time in s.
    radius : float
        Cylinder radius in mm.
    D_intra : float
        Intra-cylinder bulk diffusivity in mm^2/s.
    """
    b = np.asarray(b, dtype=float)
    # Gaussian-phase approximation: the variance of the spin phase scales
    # with the perpendicular MSD, which saturates at ~R^2/2 for long Delta.
    msd_perp = 2.0 * radius ** 2 * (
        1.0 - np.exp(-D_intra * Delta / radius ** 2)
    )
    # Effective perpendicular diffusivity at the given Delta:
    D_perp = msd_perp / (2.0 * Delta) if Delta > 0 else D_intra
    return S0 * np.exp(-b * D_perp)


def restricted_sphere_signal(
    b: NDArray[np.float64],
    Delta: float,
    radius: float,
    D_intra: float,
    S0: float = 1.0,
) -> NDArray[np.float64]:
    """Signal from water restricted inside an impermeable sphere
    (Murday--Cotts long-Delta limit).

    Same Gaussian-phase approximation as the cylinder case but with the
    isotropic MSD ceiling 6 R^2 / 5 at long Delta.
    """
    b = np.asarray(b, dtype=float)
    msd = (6.0 / 5.0) * radius ** 2 * (
        1.0 - np.exp(-D_intra * Delta / radius ** 2)
    )
    D_eff = msd / (6.0 * Delta) if Delta > 0 else D_intra
    return S0 * np.exp(-b * D_eff)


# ---------------------------------------------------------------------------
# 3. Alpha-stable displacement model
# ---------------------------------------------------------------------------

def stable_displacement_signal(
    b: NDArray[np.float64],
    alpha: float,
    D_alpha: float,
    Delta: float,
    S0: float = 1.0,
) -> NDArray[np.float64]:
    """Signal from a 3D isotropic symmetric alpha-stable displacement
    distribution (Magin, Abdullah, Baleanu, Zhou, 2008).

    The propagator P(r, Delta) is the isotropic alpha-stable density whose
    characteristic function is exp(-D_alpha Delta |q|^alpha). Hence

        S(b) = S0 * exp(-(b D_alpha)^(alpha / 2))

    in the standard b = q^2 Delta convention with q = gamma g delta.
    For alpha = 2 this reduces to Gaussian diffusion with diffusivity
    D_alpha.

    Parameters
    ----------
    alpha : float
        Stability index in (0, 2].
    D_alpha : float
        Generalised diffusion coefficient (mm^2 / s^(alpha/2) when
        alpha != 2).
    Delta : float
        Effective diffusion time, used only to make the units explicit
        (the standard b-value already absorbs Delta).
    """
    if not 0 < alpha <= 2:
        raise ValueError(f"alpha must be in (0, 2], got {alpha}")
    b = np.asarray(b, dtype=float)
    return S0 * np.exp(-((b * D_alpha) ** (alpha / 2.0)))


# ---------------------------------------------------------------------------
# 4. Multi-compartment mixtures
# ---------------------------------------------------------------------------

def multi_compartment_signal(
    b: NDArray[np.float64],
    compartments: Sequence[Dict],
    S0: float = 1.0,
) -> NDArray[np.float64]:
    """Convex combination of compartment-level signals.

    Each compartment is a dict with keys
        'fraction' : float (must sum to one across compartments)
        'kind'     : {'gaussian', 'kurtosis', 'stretched', 'cylinder',
                      'sphere', 'stable'}
        ... model-specific parameters ...
    """
    b = np.asarray(b, dtype=float)
    out = np.zeros_like(b)
    total = 0.0
    for c in compartments:
        f = float(c["fraction"])
        total += f
        kind = c["kind"]
        if kind == "gaussian":
            s = gaussian_signal(b, c["D"], S0=1.0)
        elif kind == "kurtosis":
            s = kurtosis_signal(b, c["D"], c["K"], S0=1.0)
        elif kind == "stretched":
            s = stretched_exponential_signal(b, c["D"], c["alpha_se"], S0=1.0)
        elif kind == "cylinder":
            s = restricted_cylinder_signal(
                b, c["Delta"], c["radius"], c["D_intra"], S0=1.0
            )
        elif kind == "sphere":
            s = restricted_sphere_signal(
                b, c["Delta"], c["radius"], c["D_intra"], S0=1.0
            )
        elif kind == "stable":
            s = stable_displacement_signal(
                b, c["alpha"], c["D_alpha"], c["Delta"], S0=1.0
            )
        else:
            raise ValueError(f"Unknown compartment kind: {kind}")
        out = out + f * s
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Compartment fractions sum to {total}, expected 1.")
    return S0 * out


# ---------------------------------------------------------------------------
# 5. Displacement samplers (used to verify the closed-form models and to
#    drive the empirical-CF persistence pipeline).
# ---------------------------------------------------------------------------

def sample_gaussian_displacement(
    D: float,
    Delta: float,
    n: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Sample 1D Gaussian displacements with variance 2 D Delta."""
    sigma = np.sqrt(2.0 * D * Delta)
    return rng.normal(0.0, sigma, size=n)


def sample_stable_displacement(
    alpha: float,
    D_alpha: float,
    Delta: float,
    n: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Sample 1D symmetric alpha-stable displacements.

    With the convention b = q^2 Delta and q = sqrt(b/Delta), the closed-form
    signal S(b) = exp(-(b D_alpha)^(alpha/2)) corresponds to a displacement
    characteristic function

        E[exp(i q X)] = exp(-(D_alpha Delta)^(alpha/2) |q|^alpha).

    For a symmetric alpha-stable variate in the 0-parameterisation with
    scale c, E[exp(i q X)] = exp(-c^alpha |q|^alpha), so the required scale
    is c = (D_alpha Delta)^(1/2). This recovers the Gaussian case at
    alpha = 2 with variance 2 D_alpha Delta.
    """
    scale = (D_alpha * Delta) ** 0.5
    return stable_rvs(alpha=alpha, beta=0.0, scale=scale, size=n, rng=rng)


def sample_restricted_displacement(
    radius: float,
    Delta: float,
    D_intra: float,
    n: int,
    rng: np.random.Generator,
    geometry: str = "sphere",
) -> NDArray[np.float64]:
    """Sample (approximate) restricted displacements.

    We use the Gaussian-phase approximation: displacements are zero-mean
    normal with variance equal to the geometry-dependent saturating MSD.
    This is the standard short-time/long-Delta limit used by NODDI,
    CHARMED and AxCaliber.
    """
    if geometry == "sphere":
        var = (6.0 / 5.0) * radius ** 2 * (
            1.0 - np.exp(-D_intra * Delta / radius ** 2)
        )
        sigma = np.sqrt(var / 3.0)  # per-axis variance
    elif geometry == "cylinder":
        var = 2.0 * radius ** 2 * (
            1.0 - np.exp(-D_intra * Delta / radius ** 2)
        )
        sigma = np.sqrt(var / 2.0)  # per perpendicular axis
    else:
        raise ValueError(f"Unknown geometry: {geometry}")
    return rng.normal(0.0, sigma, size=n)


def signal_from_displacements(
    b: NDArray[np.float64],
    displacements: NDArray[np.float64],
    Delta: float,
    S0: float = 1.0,
) -> NDArray[np.float64]:
    """Empirical signal from a 1D displacement sample.

    S(b) = S0 * (1/N) sum_i cos(q_i x_i), q = sqrt(b/Delta) (narrow pulse).
    Real part only because we use symmetric displacements.
    """
    b = np.asarray(b, dtype=float)
    q = np.sqrt(np.maximum(b, 0.0) / max(Delta, 1e-12))
    # broadcast: (n_b, 1) * (1, n_disp)
    phase = q[:, None] * displacements[None, :]
    return S0 * np.mean(np.cos(phase), axis=1)


def add_rician_noise(
    signal: NDArray[np.float64],
    snr: float,
    rng: np.random.Generator,
    S0: float = 1.0,
) -> NDArray[np.float64]:
    """Add Rician noise with the given SNR at b=0.

    Noise standard deviation in each Gaussian channel: sigma = S0 / SNR.
    The Rician magnitude is sqrt((s + n_r)^2 + n_i^2).
    """
    sigma = S0 / snr
    n_r = rng.normal(0.0, sigma, size=signal.shape)
    n_i = rng.normal(0.0, sigma, size=signal.shape)
    return np.sqrt((signal + n_r) ** 2 + n_i ** 2)


# ---------------------------------------------------------------------------
# 6. Standard b-shell schedules
# ---------------------------------------------------------------------------

#: Multi-shell schedule used throughout the manuscript (s/mm^2).
DEFAULT_BVALUES: Tuple[float, ...] = (
    0.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0,
)

#: Densely sampled b-grid used for persistence computation; the underlying
#: closed-form signals are smooth, so we can oversample S(b) before sublevel
#: persistence without loss of generality.
def dense_b_grid(b_max: float = 3000.0, n: int = 256) -> NDArray[np.float64]:
    """Return a dense, monotonically increasing b-grid for persistence."""
    return np.linspace(0.0, b_max, n)
