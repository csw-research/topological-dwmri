#!/usr/bin/env python3
"""Real-data (HCP) figures for the DW-MRI persistent-homology paper.

Reads per-subject NIfTI maps and ROI summaries from
``results/hcp/<subject>/`` and produces:

* fig7_hcp_maps.pdf -- representative axial slices of FA, MD, MK,
  alpha_persistence, and the tissue mask for one subject.
* fig8_hcp_roi_summary.pdf -- per-ROI (WM / GM / CSF / CC) bar charts of
  the four scalar maps, pooled across subjects.
* fig9_alpha_vs_K.pdf -- voxelwise scatter of alpha_persistence against
  MK, coloured by tissue.

Also writes results/hcp_group_summary.csv with one row per subject and
per ROI.
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

import nibabel as nib  # noqa
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Prefer the v2 layout (denoised/JHU/split-half pipeline output). Fall back to
# the older "hcp" directory if v2 isn't present.
HCP = ROOT / "results" / "hcp_v2"
if not HCP.is_dir():
    HCP = ROOT / "results" / "hcp"

# Match parent-paper figure style
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


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def load_all_summaries(hcp_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(hcp_dir.glob("*/*_roi_summary.json")):
        with open(f) as fh:
            d = json.load(fh)
        rows.append(d)
    if not rows:
        raise FileNotFoundError(f"No ROI summaries in {hcp_dir}")
    df = pd.DataFrame(rows)
    return df


def melt_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape wide JSON summary to long format with columns
    (subject, roi, metric, mean, std, n)."""
    rois = ["WM", "GM", "CSF", "CC"]
    metrics = ["FA", "MD", "MK", "alpha_persistence"]
    out = []
    for _, r in df.iterrows():
        for roi in rois:
            for m in metrics:
                key = f"{roi}_{m}_mean"
                if key not in r:
                    continue
                out.append({
                    "subject": r["subject"],
                    "roi": roi,
                    "metric": m,
                    "mean": r[key],
                    "median": r.get(f"{roi}_{m}_median", np.nan),
                    "std": r.get(f"{roi}_{m}_std", np.nan),
                    "n": r.get(f"{roi}_{m}_n", 0),
                })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Figure 7: representative maps for one subject
# ---------------------------------------------------------------------------

def fig7_subject_maps(out_dir: Path, subject: str = None) -> Path:
    candidates = sorted(p.name for p in HCP.iterdir() if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No subject directories in {HCP}")
    if subject is None:
        subject = candidates[0]
    sd = HCP / subject

    FA = nib.load(str(sd / f"{subject}_FA.nii.gz")).get_fdata()
    MD = nib.load(str(sd / f"{subject}_MD.nii.gz")).get_fdata()
    MK = nib.load(str(sd / f"{subject}_MK.nii.gz")).get_fdata()
    alpha = nib.load(str(sd / f"{subject}_alpha_persistence.nii.gz")
                     ).get_fdata()
    tissue = nib.load(str(sd / f"{subject}_tissue.nii.gz")).get_fdata()

    # axial slice through the middle
    z = FA.shape[2] // 2

    fig, axes = plt.subplots(1, 5, figsize=(10.0, 2.2))
    images = [
        ("FA", FA, "gray", (0, 1.0)),
        (r"MD ($10^{-3}$~mm$^{2}$/s)", MD * 1e3, "viridis", (0, 2.0)),
        ("MK", MK, "plasma", (0, 2.0)),
        (r"$\widehat{\alpha}_{\mathrm{persistence}}$", alpha, "cividis",
         (1.4, 2.6)),
        ("tissue", tissue, "Set2", (0, 5)),
    ]
    for ax, (name, arr, cmap, vrange) in zip(axes, images):
        sl = np.rot90(arr[:, :, z])
        im = ax.imshow(sl, cmap=cmap, vmin=vrange[0], vmax=vrange[1])
        ax.set_title(name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"HCP subject {subject} (axial midslice, downsampled 2x)",
                 fontsize=9, y=1.03)
    fig.tight_layout()
    p = out_dir / "fig7_hcp_maps.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Figure 8: ROI bar charts pooled across subjects
# ---------------------------------------------------------------------------

def fig8_roi_summary(out_dir: Path) -> Path:
    df = load_all_summaries(HCP)
    long = melt_summary(df)
    rois = ["CSF", "GM", "WM", "CC"]
    metrics = [
        ("FA",                 "FA"),
        (r"MD ($10^{-3}$~mm$^{2}$/s)", "MD"),
        ("MK",                 "MK"),
        (r"$\widehat{\alpha}_{\mathrm{persistence}}$", "alpha_persistence"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.4))
    for ax, (label, m) in zip(axes, metrics):
        sub = long[long.metric == m]
        means = []
        sems  = []
        for r in rois:
            vals = sub[sub.roi == r]["mean"].values
            if m == "MD":
                vals = vals * 1e3
            means.append(np.nanmean(vals))
            sems.append(np.nanstd(vals) / max(1, np.sqrt(len(vals))))
        x = np.arange(len(rois))
        colours = [TISSUE_COLOURS[r] for r in rois]
        ax.bar(x, means, yerr=sems, color=colours, edgecolor="black",
               linewidth=0.4, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(rois)
        ax.set_ylabel(label)
    n_subj = df["subject"].nunique()
    fig.suptitle(f"Mean $\\pm$ s.e.m. across {n_subj} HCP subjects",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    p = out_dir / "fig8_hcp_roi_summary.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Figure 9: voxelwise scatter alpha_persistence vs MK
# ---------------------------------------------------------------------------

def fig9_alpha_vs_K(out_dir: Path, max_voxels: int = 50000) -> Path:
    # Use one representative subject for the scatter
    subjects = sorted(p.name for p in HCP.iterdir() if p.is_dir())
    if not subjects:
        raise FileNotFoundError(f"No subjects in {HCP}")
    subject = subjects[0]
    sd = HCP / subject
    MK = nib.load(str(sd / f"{subject}_MK.nii.gz")).get_fdata()
    alpha = nib.load(str(sd / f"{subject}_alpha_persistence.nii.gz")
                     ).get_fdata()
    tissue = nib.load(str(sd / f"{subject}_tissue.nii.gz")).get_fdata()

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    rng = np.random.default_rng(0)
    for label_id, name in [(1, "WM"), (2, "GM"), (3, "CSF"), (4, "CC")]:
        m = (tissue == label_id) & np.isfinite(MK) & np.isfinite(alpha)
        xs = MK[m]; ys = alpha[m]
        if xs.size > max_voxels:
            idx = rng.choice(xs.size, max_voxels, replace=False)
            xs = xs[idx]; ys = ys[idx]
        ax.scatter(xs, ys, s=1.5, alpha=0.25, color=TISSUE_COLOURS[name],
                   edgecolors="none", label=name)
    ax.set_xlabel(r"MK (DKI)")
    ax.set_ylabel(r"$\widehat{\alpha}_{\mathrm{persistence}}$")
    leg = ax.legend(frameon=False, loc="lower right", markerscale=4)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    ax.set_xlim(-0.2, 2.5)
    ax.set_ylim(1.2, 3.2)
    fig.tight_layout()
    p = out_dir / "fig9_alpha_vs_K.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Group CSV
# ---------------------------------------------------------------------------

def write_group_csv(out_csv: Path) -> Path:
    df = load_all_summaries(HCP)
    long = melt_summary(df)
    long.to_csv(out_csv, index=False)
    return out_csv


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig7_subject_maps(out)
    fig8_roi_summary(out)
    fig9_alpha_vs_K(out)
    csv_path = ROOT / "results" / "hcp_group_summary.csv"
    write_group_csv(csv_path)
    print(f"wrote {csv_path}")
    for name in ["fig7_hcp_maps.pdf", "fig8_hcp_roi_summary.pdf",
                 "fig9_alpha_vs_K.pdf"]:
        print(f"wrote {out / name}")


if __name__ == "__main__":
    main()
