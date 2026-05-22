#!/usr/bin/env python3
"""True scan-rescan ICC for alpha_persistence using the HCP-YA Retest data.

Loads results/retest_v1/<subject>/<subject>_roi_summary.json and
results/retest_v2/<subject>/<subject>_roi_summary.json for each of the 44
matched retest subjects, then computes:

  * Bland-Altman summary (mean of (v1-v2) and limits of agreement)
  * Shrout-Fleiss intraclass correlation ICC(3,1)
  * Within-subject coefficient of variation (CoV)
  * Pearson r

per ROI for FA, MD, MK, alpha_persistence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "results" / "retest_v1"
V2 = ROOT / "results" / "retest_v2"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_summaries(folder: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(folder.glob("*/*_roi_summary.json")):
        with open(f) as fh:
            d = json.load(fh)
        rows.append(d)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reliability statistics
# ---------------------------------------------------------------------------

def icc_3_1(rater_a, rater_b):
    a = np.asarray(rater_a, float)
    b = np.asarray(rater_b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = a.size
    if n < 3:
        return float("nan")
    M = np.column_stack([a, b])
    BMS = M.mean(axis=1).var(ddof=1) * 2  # subject mean squares (k=2)
    WMS = ((M - M.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (2 - 1))
    if BMS + WMS == 0:
        return float("nan")
    return float((BMS - WMS) / (BMS + (2 - 1) * WMS))


def within_subject_cov(rater_a, rater_b):
    a = np.asarray(rater_a, float); b = np.asarray(rater_b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3:
        return float("nan")
    mean = 0.5 * (a + b)
    diff = a - b
    sigma_within = np.sqrt(0.5 * np.mean(diff ** 2))
    return float(sigma_within / mean.mean())


def bland_altman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    diff = a - b
    md = diff.mean()
    sd = diff.std(ddof=1)
    return {"mean_diff": float(md), "sd_diff": float(sd),
            "loa_lower": float(md - 1.96 * sd),
            "loa_upper": float(md + 1.96 * sd),
            "n": int(diff.size)}


# ---------------------------------------------------------------------------
# Per-ROI table
# ---------------------------------------------------------------------------

def reliability_table(df1, df2, rois=("WM", "GM", "CSF", "CC"),
                       metrics=("FA", "MD", "MK", "alpha_persistence")):
    out = []
    # match by subject
    df1 = df1.set_index("subject")
    df2 = df2.set_index("subject")
    common = sorted(set(df1.index) & set(df2.index))
    for roi in rois:
        for m in metrics:
            key = f"{roi}_{m}_mean"
            if key not in df1.columns or key not in df2.columns:
                continue
            a = df1.loc[common, key].values
            b = df2.loc[common, key].values
            ba = bland_altman(a, b)
            cov = within_subject_cov(a, b)
            icc = icc_3_1(a, b)
            r, _ = pearsonr(
                a[np.isfinite(a) & np.isfinite(b)],
                b[np.isfinite(a) & np.isfinite(b)],
            ) if np.sum(np.isfinite(a) & np.isfinite(b)) > 2 else (np.nan, np.nan)
            out.append({
                "roi": roi, "metric": m, "n_subjects": ba["n"],
                "ICC_3_1": icc, "within_subject_CoV": cov,
                "pearson_r": float(r),
                "mean_v1": float(np.nanmean(a)),
                "mean_v2": float(np.nanmean(b)),
                "BA_mean_diff": ba["mean_diff"],
                "BA_loa_lower": ba["loa_lower"],
                "BA_loa_upper": ba["loa_upper"],
            })
    return pd.DataFrame(out)


def main():
    df1 = load_summaries(V1)
    df2 = load_summaries(V2)
    print(f"Loaded V1: {len(df1)} subjects, V2: {len(df2)} subjects")
    table = reliability_table(df1, df2)
    out_csv = ROOT / "results" / "retest_reliability.csv"
    table.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}\n")

    # Pretty-print alpha_persistence rows
    ap = table[table.metric == "alpha_persistence"]
    print("=== Scan-rescan reliability of alpha_persistence (44 HCP subjects) ===")
    print(ap.to_string(index=False))
    print()
    print("=== All metrics, ICC summary ===")
    piv = table.pivot(index="roi", columns="metric", values="ICC_3_1")
    print(piv.round(3))


if __name__ == "__main__":
    main()
