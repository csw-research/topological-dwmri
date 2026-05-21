#!/usr/bin/env python3
"""Additional figures for the v2 analysis: test-retest scatter, JHU tract
panel."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HCP = ROOT / "results" / "hcp_v2"

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

TISSUE_COLOURS = {
    "WM":  "#4477AA",
    "GM":  "#228833",
    "CSF": "#66CCEE",
    "CC":  "#EE6677",
}


def load_long():
    rows = []
    for f in sorted(HCP.glob("*/*_roi_summary.json")):
        with open(f) as fh:
            rows.append(json.load(fh))
    long = []
    rois = ["WM", "GM", "CSF", "CC"]
    metrics = ["FA", "MD", "MK",
               "alpha_persistence", "alpha_split_A", "alpha_split_B"]
    for r in rows:
        for roi in rois:
            for m in metrics:
                k = f"{roi}_{m}_mean"
                if k in r:
                    long.append({"subject": r["subject"], "roi": roi,
                                  "metric": m, "mean": r[k]})
    return pd.DataFrame(long)


# ---------------------------------------------------------------------------
# Figure 10: split-half test-retest scatter
# ---------------------------------------------------------------------------

def fig10_split_half(out_dir: Path) -> Path:
    long = load_long()
    icc_df = pd.read_csv(ROOT / "results" / "hcp_v2_split_reliability.csv")
    icc = {row["roi"]: row["ICC_3_1"] for _, row in icc_df.iterrows()}

    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    for roi in ["WM", "GM", "CSF", "CC"]:
        a = long[(long.roi == roi) & (long.metric == "alpha_split_A")
                 ].sort_values("subject")["mean"].values
        b = long[(long.roi == roi) & (long.metric == "alpha_split_B")
                 ].sort_values("subject")["mean"].values
        ax.scatter(a, b, s=18, color=TISSUE_COLOURS[roi],
                   edgecolors="black", linewidths=0.4,
                   label=f"{roi}  (ICC = {icc[roi]:.2f})")
    lo, hi = 1.5, 2.6
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.5, alpha=0.6)
    ax.set_xlabel(r"$\widehat{\alpha}_{\mathrm{persistence}}$, split A")
    ax.set_ylabel(r"$\widehat{\alpha}_{\mathrm{persistence}}$, split B")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    p = out_dir / "fig10_split_half.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Figure 11: JHU tract panel
# ---------------------------------------------------------------------------

def fig11_jhu_tracts(out_dir: Path, top_n: int = 15) -> Path:
    df = pd.read_csv(ROOT / "results" / "hcp_v2_jhu_group.csv")
    # Drop tracts with <20 subjects (boundary clipping artefacts)
    df = df[df["count"] >= 20]
    ap = df[df.metric == "alpha_persistence"].sort_values("mean",
                                                            ascending=False)
    ap = ap.head(top_n)
    mk = df[df.metric == "MK"].set_index("label").reindex(ap.label.tolist())

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.5),
                             gridspec_kw={"width_ratios": [1.2, 1.2]})
    y = np.arange(len(ap))
    # Strip the leading "NN_" index from labels for readability
    def clean(lbl):
        if lbl[:3].endswith("_"):
            lbl = lbl[3:]
        return lbl.replace("_", " ")
    labels = [clean(l) for l in ap.label.values]

    axes[0].barh(y, ap["mean"].values, xerr=ap["std"].values,
                 color="#4477AA", edgecolor="black", linewidth=0.4,
                 capsize=2)
    axes[0].set_yticks(y); axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel(r"$\widehat{\alpha}_{\mathrm{persistence}}$")
    axes[0].set_title(f"(a)  Top {top_n} JHU tracts (N=30)")
    axes[0].set_xlim(2.0, 2.55)

    axes[1].barh(y, mk["mean"].values, xerr=mk["std"].values,
                 color="#EE6677", edgecolor="black", linewidth=0.4,
                 capsize=2)
    axes[1].set_yticks(y); axes[1].set_yticklabels([""] * len(y))
    axes[1].invert_yaxis()
    axes[1].set_xlabel("MK")
    axes[1].set_title("(b)  MK in the same tracts")
    axes[1].set_xlim(0.85, 1.25)

    fig.tight_layout()
    p = out_dir / "fig11_jhu_tracts.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    print("wrote", fig10_split_half(out))
    print("wrote", fig11_jhu_tracts(out))


if __name__ == "__main__":
    main()
