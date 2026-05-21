#!/usr/bin/env python3
"""Publication-quality figures for the DW-MRI persistent-homology paper.

The visual style matches the parent paper (Times serif, Tol's bright
colour palette, 300 dpi PDF outputs).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.5,
    "lines.linewidth": 1.0,
})

COLORS = {
    0.5: "#EE6677",
    1.0: "#228833",
    1.2: "#4477AA",
    1.5: "#CCBB44",
    1.8: "#66CCEE",
    2.0: "#AA3377",
    "free":       "#228833",
    "restricted": "#EE6677",
    "stable":     "#4477AA",
    "mix":        "#AA3377",
}


def _load(name: str) -> dict:
    return json.load(open(ROOT / "results" / f"{name}.json"))


# ---------------------------------------------------------------------------
# Figure 1: parent-paper benchmark (Experiment 1)
# ---------------------------------------------------------------------------

def fig1_parent_benchmark(out_dir: Path) -> Path:
    d = _load("experiment_1_parent_replication")
    alphas = np.asarray(d["alphas"])
    ah = np.asarray(d["alpha_hats"])
    lo = np.asarray(d["alpha_ci_low"])
    hi = np.asarray(d["alpha_ci_high"])

    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.plot([0.9, 2.1], [0.9, 2.1], "k--", linewidth=0.5, alpha=0.6,
            label="identity")
    ax.errorbar(
        alphas, ah, yerr=[ah - lo, hi - ah],
        fmt="o", color="#4477AA", capsize=2, markersize=5,
        linewidth=0.9, label=r"$\widehat{\alpha}_{\mathrm{tail}}$",
    )
    ax.set_xlabel(r"$\alpha$ (true)")
    ax.set_ylabel(r"$\widehat{\alpha}_{\mathrm{tail}}$")
    ax.set_xlim(0.9, 2.1)
    ax.set_ylim(0.9, 2.2)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    p = out_dir / "fig1_parent_benchmark.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Figure 2: representative DW-MRI signals and persistence diagrams
# ---------------------------------------------------------------------------

def fig2_signals_and_diagrams(out_dir: Path) -> Path:
    sys.path.insert(0, str(ROOT))
    from src.generators.dwmri_signal import (
        gaussian_signal, kurtosis_signal,
        stretched_exponential_signal, stable_displacement_signal,
        DEFAULT_BVALUES,
    )
    from src.utils.parent_bridge import (
        sublevel_persistence_1d, persistence_lifetimes,
        stable_levy_process,
    )

    b_dense = np.linspace(0, 3000, 200)
    b_acq = np.asarray(DEFAULT_BVALUES)
    D = 0.8e-3

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))

    # Panel (a): S(b) for several models
    ax = axes[0]
    ax.plot(b_dense, gaussian_signal(b_dense, D),
            color=COLORS[2.0], linewidth=1.2, label="Gaussian")
    ax.plot(b_dense, kurtosis_signal(b_dense, D, K=1.0),
            color="#888888", linewidth=1.2, label=r"DKI $K=1$")
    ax.plot(b_dense, stable_displacement_signal(b_dense, 1.5, D, 0.05),
            color=COLORS[1.5], linewidth=1.2, label=r"stable $\alpha=1.5$")
    ax.plot(b_dense, stable_displacement_signal(b_dense, 1.2, D, 0.05),
            color=COLORS[1.2], linewidth=1.2, label=r"stable $\alpha=1.2$")
    ax.plot(b_acq, gaussian_signal(b_acq, D), "o", color=COLORS[2.0],
            markersize=2.5)
    ax.set_xlabel(r"$b$ (s/mm$^2$)")
    ax.set_ylabel(r"$S(b)/S_0$")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("(a)")

    # Panel (b): Levy-bridge sample path under each alpha
    ax = axes[1]
    rng = np.random.default_rng(7)
    for alpha in (2.0, 1.5, 1.2):
        _, path = stable_levy_process(
            alpha=alpha, n_steps=500, dt=1.0 / 500, d=1, rng=rng,
        )
        ax.plot(np.arange(path.shape[0]), path[:, 0], color=COLORS[alpha],
                linewidth=0.7, label=fr"$\alpha={alpha}$")
    ax.set_xlabel("step index")
    ax.set_ylabel("$L_t$")
    ax.legend(frameon=False, loc="best")
    ax.set_title("(b)")

    # Panel (c): persistence diagram of one stable path per alpha
    ax = axes[2]
    rng = np.random.default_rng(11)
    for alpha in (2.0, 1.5, 1.2):
        _, path = stable_levy_process(
            alpha=alpha, n_steps=2000, dt=1.0 / 2000, d=1, rng=rng,
        )
        diag = sublevel_persistence_1d(path[:, 0])
        if len(diag) > 0:
            ax.scatter(diag[:, 0], diag[:, 1], s=3, alpha=0.5,
                       color=COLORS[alpha], edgecolors="none",
                       label=fr"$\alpha={alpha}$")
    lims = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(lims), max(lims)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.3, alpha=0.6)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.legend(frameon=False, loc="best")
    ax.set_title("(c)")

    fig.tight_layout()
    p = out_dir / "fig2_signals_and_diagrams.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Figure 3: kurtosis / stretched-alpha bias-variance (Experiment 2)
# ---------------------------------------------------------------------------

def fig3_signal_model_estimators(out_dir: Path) -> Path:
    d = _load("experiment_2_signal_models")
    names = list(d["results"].keys())
    K_means = []
    K_stds = []
    a_means = []
    a_stds = []
    labels = []
    for n in names:
        Kh = np.asarray(d["results"][n]["K_hat"])
        ah = np.asarray(d["results"][n]["alpha_se_hat"])
        K_means.append(np.nanmedian(Kh))
        K_stds.append(np.nanstd(Kh))
        a_means.append(np.nanmedian(ah))
        a_stds.append(np.nanstd(ah))
        labels.append(n.replace("_", " "))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    x = np.arange(len(names))
    axes[0].bar(x, K_means, yerr=K_stds, color="#4477AA",
                edgecolor="black", linewidth=0.4, capsize=2)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylabel(r"$\widehat{K}$ (DKI)")
    axes[0].axhline(0.0, color="k", linewidth=0.3)
    axes[0].set_title("(a)")

    axes[1].bar(x, a_means, yerr=a_stds, color="#EE6677",
                edgecolor="black", linewidth=0.4, capsize=2)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_ylabel(r"$\widehat{\alpha}_{\mathrm{se}}$ (stretched)")
    axes[1].axhline(2.0, color="k", linewidth=0.3, linestyle=":")
    axes[1].set_title("(b)")

    fig.tight_layout()
    p = out_dir / "fig3_estimator_bias.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Figure 4: region pooling per microstructure (Experiment 3)
# ---------------------------------------------------------------------------

def fig4_region_pooling(out_dir: Path) -> Path:
    d = _load("experiment_3_region_pooling")
    rows = d["diagnostics"]
    labels = []
    for r in rows:
        if r["kind"] == "stable":
            labels.append(f"stable $\\alpha={r['params']['alpha']}$")
        else:
            labels.append(r["kind"])

    K = np.array([r["mean_K"] for r in rows])
    K_std = np.array([r["std_K"] for r in rows])
    ase = np.array([r["mean_alpha_se"] for r in rows])
    ase_std = np.array([r["std_alpha_se"] for r in rows])
    ap = np.array([r["alpha_cumulative"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))
    x = np.arange(len(rows))

    axes[0].bar(x, K, yerr=K_std, color="#4477AA", edgecolor="black",
                linewidth=0.4, capsize=2)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylabel(r"mean $\widehat{K}$")
    axes[0].set_title("(a) DKI kurtosis")

    axes[1].bar(x, ase, yerr=ase_std, color="#EE6677", edgecolor="black",
                linewidth=0.4, capsize=2)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_ylabel(r"mean $\widehat{\alpha}_{\mathrm{se}}$")
    axes[1].set_ylim(0.5, 2.1)
    axes[1].set_title("(b) stretched exponent")

    axes[2].bar(x, ap, color="#CCBB44", edgecolor="black", linewidth=0.4)
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, rotation=30, ha="right")
    axes[2].set_ylabel(r"$\widehat{\alpha}_{\mathrm{persistence}}$")
    axes[2].set_title("(c) persistence-tail")

    fig.tight_layout()
    p = out_dir / "fig4_region_pooling.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Figure 5: calibration curves (Experiment 4)
# ---------------------------------------------------------------------------

def fig5_calibration(out_dir: Path) -> Path:
    d = _load("experiment_4_calibration")
    rows = d["rows"]
    a_true = np.array([r["alpha_true"] for r in rows])
    K_mean = np.array([r["K_mean"] for r in rows])
    K_std  = np.array([r["K_std"] for r in rows])
    ase_mean = np.array([r["alpha_se_mean"] for r in rows])
    ase_std  = np.array([r["alpha_se_std"] for r in rows])
    aph = np.array([r["alpha_persistence"] for r in rows])
    aph_lo = np.array([r["alpha_persistence_lo"] for r in rows])
    aph_hi = np.array([r["alpha_persistence_hi"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))

    axes[0].errorbar(a_true, K_mean, yerr=K_std, fmt="o-",
                     color="#4477AA", markersize=4, linewidth=0.8,
                     capsize=2)
    axes[0].set_xlabel(r"true $\alpha$")
    axes[0].set_ylabel(r"mean $\widehat{K}$")
    axes[0].set_title("(a)")
    axes[0].invert_xaxis()

    axes[1].errorbar(a_true, ase_mean, yerr=ase_std, fmt="s-",
                     color="#EE6677", markersize=4, linewidth=0.8,
                     capsize=2)
    axes[1].set_xlabel(r"true $\alpha$")
    axes[1].set_ylabel(r"mean $\widehat{\alpha}_{\mathrm{se}}$")
    axes[1].set_title("(b)")
    axes[1].invert_xaxis()

    axes[2].errorbar(
        a_true, aph, yerr=[aph - aph_lo, aph_hi - aph], fmt="D-",
        color="#CCBB44", markersize=4, linewidth=0.8, capsize=2,
    )
    axes[2].plot([1.0, 2.0], [1.0, 2.0], "k--", linewidth=0.4, alpha=0.6)
    axes[2].set_xlabel(r"true $\alpha$")
    axes[2].set_ylabel(r"$\widehat{\alpha}_{\mathrm{persistence}}$")
    axes[2].set_title("(c)")
    axes[2].invert_xaxis()

    fig.tight_layout()
    p = out_dir / "fig5_calibration.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Figure 6: schematic / pipeline (no data, conceptual)
# ---------------------------------------------------------------------------

def fig6_pipeline(out_dir: Path) -> Path:
    """Single-panel conceptual diagram of the pipeline."""
    fig, ax = plt.subplots(figsize=(7.0, 2.0))
    ax.axis("off")
    boxes = [
        ("DW-MRI\n(voxels, shells, dirs)", 0.05, 0.5),
        (r"attenuation $-\ln S/S_0$", 0.27, 0.5),
        ("cumulative\nLevy bridge", 0.47, 0.5),
        ("sublevel\npersistence", 0.66, 0.5),
        (r"$\widehat{\alpha}_{\mathrm{tail}}$ via Hill", 0.86, 0.5),
    ]
    for txt, x, y in boxes:
        ax.add_patch(plt.Rectangle((x - 0.07, y - 0.18), 0.14, 0.36,
                                   facecolor="#EEEEEE", edgecolor="black",
                                   linewidth=0.5))
        ax.text(x, y, txt, ha="center", va="center", fontsize=8)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][1] + 0.07
        x1 = boxes[i + 1][1] - 0.07
        ax.annotate("", xy=(x1, 0.5), xytext=(x0, 0.5),
                    arrowprops=dict(arrowstyle="->", linewidth=0.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    p = out_dir / "fig6_pipeline.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for fn in (fig6_pipeline, fig1_parent_benchmark, fig2_signals_and_diagrams,
               fig3_signal_model_estimators, fig4_region_pooling,
               fig5_calibration):
        try:
            p = fn(out)
            print(f"wrote {p}")
        except Exception as e:  # pragma: no cover - debugging aid
            print(f"FAILED {fn.__name__}: {e!r}")


if __name__ == "__main__":
    main()
