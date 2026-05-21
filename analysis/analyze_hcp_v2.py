#!/usr/bin/env python3
"""Statistical analysis of the v2 HCP pipeline output.

Computes:
  1. Tissue-class ANOVA + permutation-based pairwise contrasts.
  2. Split-half test-retest reliability:
        intraclass correlation coefficient ICC(3,1),
        within-subject coefficient of variation CoV,
        Bland-Altman agreement.
  3. JHU per-tract group statistics (mean +/- s.d. across subjects).

Outputs:
  results/hcp_v2_tissue_perm.csv   - per-pair permutation p-values
  results/hcp_v2_split_reliability.csv
  results/hcp_v2_jhu_group.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, spearmanr, pearsonr

ROOT = Path(__file__).resolve().parents[1]
HCP = ROOT / "results" / "hcp_v2"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_summaries():
    rows = []
    for f in sorted(HCP.glob("*/*_roi_summary.json")):
        with open(f) as fh:
            rows.append(json.load(fh))
    return rows


def tissue_long(summaries):
    rois = ["WM", "GM", "CSF", "CC"]
    metrics = ["FA", "MD", "MK",
               "alpha_persistence", "alpha_split_A", "alpha_split_B"]
    out = []
    for r in summaries:
        for roi in rois:
            for m in metrics:
                key = f"{roi}_{m}_mean"
                if key not in r:
                    continue
                out.append({
                    "subject": r["subject"], "roi": roi, "metric": m,
                    "mean": r[key],
                    "median": r.get(f"{roi}_{m}_median", np.nan),
                    "std": r.get(f"{roi}_{m}_std", np.nan),
                    "n": r.get(f"{roi}_{m}_n", 0),
                })
    return pd.DataFrame(out)


def jhu_long(summaries):
    rows = []
    for r in summaries:
        subj = r["subject"]
        for key, val in r.items():
            if not key.startswith("JHU_"):
                continue
            # parse e.g. "JHU_05_Genu_of_corpus_callosum_alpha_persistence_mean"
            parts = key.split("_")
            try:
                # find which metric suffix this is
                for sfx, m in [
                    ("_FA_mean", "FA"),
                    ("_MD_mean", "MD"),
                    ("_MK_mean", "MK"),
                    ("_alpha_persistence_mean", "alpha_persistence"),
                    ("_alpha_split_A_mean", "alpha_split_A"),
                    ("_alpha_split_B_mean", "alpha_split_B"),
                ]:
                    if key.endswith(sfx):
                        label = key[len("JHU_"):-len(sfx)]
                        rows.append({
                            "subject": subj, "label": label,
                            "metric": m, "mean": val,
                        })
                        break
            except Exception:
                continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def permutation_paired(a, b, n_perm=10000, rng=None):
    """Two-sided paired permutation test on the mean difference."""
    if rng is None:
        rng = np.random.default_rng(42)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3:
        return np.nan, np.nan
    diff = a - b
    obs = np.mean(diff)
    flips = rng.choice([-1.0, 1.0], size=(n_perm, diff.size))
    null = np.mean(flips * diff, axis=1)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return float(obs), float(p)


def icc_3_1(rater_a, rater_b):
    """Intraclass correlation ICC(3,1) (two-way mixed, single rater,
    consistency). Standard Shrout-Fleiss formula."""
    a = np.asarray(rater_a, float)
    b = np.asarray(rater_b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = a.size
    if n < 3:
        return float("nan")
    M = np.column_stack([a, b])
    grand = M.mean()
    BMS = M.mean(axis=1).var(ddof=1) * 2  # subject mean squares (k=2)
    # within-subject mean square
    WMS = ((M - M.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (2 - 1))
    if BMS + WMS == 0:
        return float("nan")
    return float((BMS - WMS) / (BMS + (2 - 1) * WMS))


def within_subject_cov(rater_a, rater_b):
    """Within-subject coefficient of variation = sigma_within / mean."""
    a = np.asarray(rater_a, float); b = np.asarray(rater_b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3:
        return float("nan")
    mean = 0.5 * (a + b)
    diff = a - b
    sigma_within = np.sqrt(0.5 * np.mean(diff ** 2))
    return float(sigma_within / mean.mean())


# ---------------------------------------------------------------------------
# Main analyses
# ---------------------------------------------------------------------------

def analyse_tissue(summaries):
    long = tissue_long(summaries)

    # ANOVA across ROIs on alpha_persistence
    groups = []
    for r in ["CSF", "GM", "WM", "CC"]:
        v = long[(long.roi == r) & (long.metric == "alpha_persistence")
                 ]["mean"].values
        groups.append(v)
    F, p_anova = f_oneway(*groups)
    print(f"ANOVA alpha_persistence across CSF/GM/WM/CC: "
          f"F = {F:.2f}, parametric p = {p_anova:.3e}")

    rows = []
    for ra, rb in [("WM", "GM"), ("CSF", "GM"), ("CC", "WM"),
                    ("CSF", "CC"), ("CC", "GM")]:
        a = long[(long.roi == ra) & (long.metric == "alpha_persistence")
                 ].sort_values("subject")["mean"].values
        b = long[(long.roi == rb) & (long.metric == "alpha_persistence")
                 ].sort_values("subject")["mean"].values
        obs, p = permutation_paired(a, b, n_perm=10000)
        rows.append({"contrast": f"{ra} vs {rb}", "mean_diff": obs,
                     "perm_p": p, "n": len(a)})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "hcp_v2_tissue_perm.csv", index=False)
    print("\n=== Pairwise paired permutation tests (10,000 perms) ===")
    print(df.to_string(index=False))
    return long


def analyse_reliability(long):
    rows = []
    for roi in ["WM", "GM", "CSF", "CC"]:
        a = long[(long.roi == roi) & (long.metric == "alpha_split_A")
                 ].sort_values("subject")["mean"].values
        b = long[(long.roi == roi) & (long.metric == "alpha_split_B")
                 ].sort_values("subject")["mean"].values
        if len(a) == 0 or len(b) == 0:
            continue
        icc = icc_3_1(a, b)
        cov = within_subject_cov(a, b)
        r_p, _ = pearsonr(a, b)
        # mean absolute difference
        mad = float(np.mean(np.abs(a - b)))
        rows.append({"roi": roi, "n_subjects": len(a),
                     "ICC_3_1": icc, "within_subject_CoV": cov,
                     "pearson_r": r_p, "mean_abs_diff": mad,
                     "mean_split_A": float(np.mean(a)),
                     "mean_split_B": float(np.mean(b))})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "hcp_v2_split_reliability.csv",
              index=False)
    print("\n=== Split-half test-retest reliability of alpha_persistence ===")
    print(df.to_string(index=False))
    return df


def analyse_jhu(summaries):
    long = jhu_long(summaries)
    if long.empty:
        print("No JHU summaries found.")
        return long
    # group statistics per label x metric
    g = long.groupby(["label", "metric"])["mean"].agg(["mean", "std",
                                                          "count"]).reset_index()
    g.to_csv(ROOT / "results" / "hcp_v2_jhu_group.csv", index=False)
    # Focus on alpha_persistence and MK
    print("\n=== Top JHU labels by mean alpha_persistence (N=30) ===")
    ap = g[g.metric == "alpha_persistence"].sort_values("mean",
                                                         ascending=False)
    print(ap.head(15).to_string(index=False))
    print("\n=== Top JHU labels by mean MK ===")
    mk = g[g.metric == "MK"].sort_values("mean", ascending=False)
    print(mk.head(15).to_string(index=False))

    # Per-tract Spearman between alpha and MK across subjects
    pairs = []
    for label in long.label.unique():
        a = long[(long.label == label)
                  & (long.metric == "alpha_persistence")
                  ].sort_values("subject")["mean"].values
        m = long[(long.label == label) & (long.metric == "MK")
                  ].sort_values("subject")["mean"].values
        if len(a) < 5:
            continue
        rho, p = spearmanr(a, m)
        pairs.append({"label": label, "n": len(a), "rho": rho, "p": p})
    pdf = pd.DataFrame(pairs).sort_values("rho", ascending=False)
    print("\n=== Per-tract Spearman(alpha, MK) across subjects ===")
    print(pdf.head(10).to_string(index=False))
    return long, pdf


def main():
    summaries = load_summaries()
    print(f"Loaded {len(summaries)} subject summaries from {HCP}")
    long_tissue = analyse_tissue(summaries)
    analyse_reliability(long_tissue)
    analyse_jhu(summaries)


if __name__ == "__main__":
    main()
