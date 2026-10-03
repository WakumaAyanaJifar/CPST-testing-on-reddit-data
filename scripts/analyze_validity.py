#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper 1, validity analyses
==========================

Four additions that turn a descriptive analysis into a measurement paper.

  1. SPLIT-HALF RELIABILITY
     Every correlation is bounded by the square root of the product of the
     reliabilities of the two measures. A block of noisy features cannot cohere
     whatever it measures, so the physical space's failure to cohere means
     nothing until reliability is known. Requires the two half-sample feature
     files produced by extract_cpst_features.py --half 1 and --half 2.

  2. IS THE CPST PARTITION BETTER THAN AN ARBITRARY ONE?
     Within-block association exceeding between-block association is not by
     itself evidence for THIS partition. Two comparisons:
       (a) against 1,000 random relabellings preserving block sizes,
       (b) against the partition the correlation structure produces on its own,
           compared by adjusted Rand index.
     If (a) fails, the assignment carries no information. If (b) is near zero,
     the data organise themselves differently from the framework, which is a
     finding rather than a failure.

  3. KNOWN-GROUPS VALIDITY
     Two contrasts the corpus supports and the main analysis never used.
       Seed vs control on matched pairs, using CYBER, PHYSICAL and SOCIAL only.
         The thinking space is excluded because seeds were selected on their
         text; including it would be circular by construction.
       Anxiety vs disengagement block, non-circular for all four spaces except
         the two community-share features, which are excluded.

  4. REDUNDANCY
     A canonical correlation says two blocks share a direction. It does not say
     how much variance. Reported as the mean squared cross-loading of one
     block's features on the other block's first canonical variate.

Requires: pandas numpy scipy matplotlib scikit-learn
    python -m pip install scikit-learn

Usage
    python extract_cpst_features.py --half 1
    python extract_cpst_features.py --half 2
    python analyze_validity.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, adjusted_rand_score
    from sklearn.model_selection import StratifiedKFold
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError:          # scikit-learn < 1.0
        StratifiedGroupKFold = None
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
except ImportError:
    sys.exit("scikit-learn is required:  python -m pip install scikit-learn")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# Data folder: set CPST_BASE_DIR to override the default location.
BASE_DIR = os.environ.get(
    "CPST_BASE_DIR",
    os.path.join(os.path.expanduser("~"), "Documents", "CPST_FOMO_v6"))
OUT_DIR = os.path.join(BASE_DIR, "paper1")

BLOCKS = ["C", "P", "S", "T"]
BLOCK_NAMES = {"C": "Cyber", "P": "Physical", "S": "Social", "T": "Thinking"}
BLOCK_COLOR = {"C": "#2a78d6", "P": "#eb6834", "S": "#1baf7a", "T": "#4a3aa7"}
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8985"
GRID, SURFACE = "#e6e5e1", "#ffffff"
COL_1, COL_2 = 3.5, 7.16

NON_FEATURES = {"author_hash", "cohort", "seed_block", "n_items", "tenure_days",
                "first_seen", "last_seen", "matched", "pair_id"}
# P_utc_offset_est is the LOCATION PARAMETER used to derive the local-time features, not a measurement of anyone's physical state. Left in the feature matrix it injects geography, and therefore language background, into the Physical block.
DROP_FEATURES = {"S_comment_share", "C_offblock_share", "T_affect_balance",
                 "T_temporal_balance", "T_self_other_ratio", "S_max_depth",
                 "P_utc_offset_est"}
# excluded from the community contrast: they define the groups
COMMUNITY_DEFINING = {"C_anxiety_share", "C_diseng_share"}

# Excluded from the seed/control contrast: they encode the sampling design.
# Seed authors were found by their SUBMISSIONS matching a keyword; controls were
# drawn from thread COMMENTERS. Any feature recording the submission/comment mix
# therefore separates the cohorts by construction, not by validity.
SELECTION_CONFOUND = {"C_post_share", "S_reply_to_comment_share",
                      "S_items_per_thread", "S_n_threads"}


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.size": 8,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "axes.edgecolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.labelcolor": INK,
        "grid.color": GRID, "grid.linewidth": 0.5,
        "lines.linewidth": 2.0, "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"),
                    dpi=600 if ext == "png" else None)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def block_of(c): return c.split("_", 1)[0]


def rank_z(df):
    r = df.rank(method="average")
    return (r - r.mean()) / r.std(ddof=1)


def residualise(df, covars):
    Z = np.column_stack([np.ones(len(covars))] +
                        [covars[c].to_numpy() for c in covars.columns])
    out = df.copy()
    for col in df.columns:
        y = df[col].to_numpy(float)
        ok = np.isfinite(y)
        if ok.sum() < 50:
            continue
        beta, *_ = np.linalg.lstsq(Z[ok], y[ok], rcond=None)
        res = np.full(len(y), np.nan)
        res[ok] = y[ok] - Z[ok] @ beta
        out[col] = res
    return out


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


# ======================================================================
# 1. split-half reliability
# ======================================================================

def _residualise_half(h: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    """Residualise one half on ITS OWN log record count and log tenure.

    The separability analysis works on volume-residualised features, so the
    reliabilities used to disattenuate it must be reliabilities of the same
    residualised quantities. Raw-feature reliability is inflated by volume,
    which is itself almost perfectly reliable across alternating halves.
    """
    cols = [c for c in feat_cols if c in h.columns]
    cov = pd.DataFrame({"log_n": np.log1p(h.n_items.to_numpy(float)),
                        "log_t": np.log1p(h.tenure_days.to_numpy(float))},
                       index=h.index)
    out = h.copy()
    out[cols] = residualise(h[cols], cov)
    return out


def reliability(h1: pd.DataFrame, h2: pd.DataFrame,
                feat_cols: List[str]) -> pd.DataFrame:
    """Split-half reliability of the RESIDUALISED features (primary, column
    r_sb) and of the raw features (r_sb_raw, reported for comparison)."""
    common = sorted(set(h1.author_hash) & set(h2.author_hash))
    print(f"  authors present in both halves: {len(common):,}")
    raw = _reliability_core(h1, h2, feat_cols, common)
    res = _reliability_core(_residualise_half(h1, feat_cols),
                            _residualise_half(h2, feat_cols), feat_cols, common)
    res = res.merge(raw[["feature", "r_half", "r_sb"]].rename(
        columns={"r_half": "r_half_raw", "r_sb": "r_sb_raw"}),
        on="feature", how="left")
    big = res[(res.r_sb_raw - res.r_sb) > 0.10]
    if len(big):
        print("  features whose reliability drops by >0.10 once volume is removed:")
        for r in big.itertuples():
            print(f"    {r.feature:<32} raw {r.r_sb_raw:.3f} -> residual {r.r_sb:.3f}")
    return res


def _reliability_core(h1: pd.DataFrame, h2: pd.DataFrame,
                      feat_cols: List[str], common: List[str]) -> pd.DataFrame:
    a = h1.set_index("author_hash").loc[common]
    b = h2.set_index("author_hash").loc[common]

    rows = []
    for col in feat_cols:
        if col not in a.columns or col not in b.columns:
            continue
        x, y = a[col].to_numpy(float), b[col].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 50:
            rows.append({"feature": col, "block": block_of(col), "n": int(ok.sum()),
                         "r_half": np.nan, "r_sb": np.nan})
            continue
        r = stats.spearmanr(x[ok], y[ok]).statistic
        # Spearman-Brown: reliability of the full-length measure
        sb = (2 * r) / (1 + r) if r > -1 else np.nan
        rows.append({"feature": col, "block": block_of(col), "n": int(ok.sum()),
                     "r_half": float(r), "r_sb": float(min(sb, 1.0))})
    return pd.DataFrame(rows)


def disattenuated_separability(corr: pd.DataFrame, cols: List[str],
                               rel: pd.DataFrame) -> Dict:
    """Within- and between-block association corrected for unreliability.

    An observed correlation is bounded by the square root of the product of the
    two measures' reliabilities. Without this correction, a block of noisy
    features looks incoherent whether or not it measures one thing, and the
    distinction between "these are independent constructs" and "these are badly
    measured" cannot be drawn.
    """
    if not len(rel):
        return {}
    relmap = dict(zip(rel.feature, rel.r_sb))
    out = {}
    for b in BLOCKS:
        cb = [c for c in cols if block_of(c) == b]
        w_obs, w_dis = [], []
        for i, j in combinations(cb, 2):
            r = abs(corr.loc[i, j])
            if not np.isfinite(r):
                continue
            w_obs.append(r)
            ri, rj = relmap.get(i, np.nan), relmap.get(j, np.nan)
            if np.isfinite(ri) and np.isfinite(rj) and ri > 0 and rj > 0:
                w_dis.append(min(r / np.sqrt(ri * rj), 1.0))
        o_obs, o_dis = [], []
        for i in cb:
            for j in cols:
                if block_of(j) == b:
                    continue
                r = abs(corr.loc[i, j])
                if not np.isfinite(r):
                    continue
                o_obs.append(r)
                ri, rj = relmap.get(i, np.nan), relmap.get(j, np.nan)
                if np.isfinite(ri) and np.isfinite(rj) and ri > 0 and rj > 0:
                    o_dis.append(min(r / np.sqrt(ri * rj), 1.0))
        out[b] = {
            "rel_median": float(rel[rel.block == b].r_sb.median()),
            "within_obs": float(np.median(w_obs)) if w_obs else np.nan,
            "within_dis": float(np.median(w_dis)) if w_dis else np.nan,
            "between_obs": float(np.median(o_obs)) if o_obs else np.nan,
            "between_dis": float(np.median(o_dis)) if o_dis else np.nan,
        }
        out[b]["gap_obs"] = out[b]["within_obs"] - out[b]["between_obs"]
        out[b]["gap_dis"] = out[b]["within_dis"] - out[b]["between_dis"]
    return out


def disattenuate(obs_r: float, rel_x: float, rel_y: float) -> float:
    """Correlation corrected for unreliability in both measures."""
    d = np.sqrt(max(rel_x, 1e-9) * max(rel_y, 1e-9))
    return float(np.clip(obs_r / d, -1, 1)) if d > 0 else np.nan


# ======================================================================
# 2. partition comparison
# ======================================================================

def separability_margin(corr: pd.DataFrame, labels: Dict[str, str],
                        cols: List[str]) -> float:
    by_block: Dict[str, List[str]] = {}
    for c in cols:
        by_block.setdefault(labels[c], []).append(c)
    within, between = [], []
    for b, cb in by_block.items():
        within += [abs(corr.loc[i, j]) for i, j in combinations(cb, 2)]
        between += [abs(corr.loc[i, j]) for i in cb for j in cols
                    if labels[j] != b]
    within = [v for v in within if np.isfinite(v)]
    between = [v for v in between if np.isfinite(v)]
    if not within or not between:
        return np.nan
    return float(np.median(within) - np.median(between))


def partition_tests(corr: pd.DataFrame, cols: List[str], n_perm: int,
                    rng, X: pd.DataFrame | None = None) -> Dict:
    true_labels = {c: block_of(c) for c in cols}
    obs = separability_margin(corr, true_labels, cols)

    sizes = [sum(1 for c in cols if block_of(c) == b) for b in BLOCKS]
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = list(cols)
        rng.shuffle(perm)
        lab, k = {}, 0
        for b, n in zip(BLOCKS, sizes):
            for c in perm[k:k + n]:
                lab[c] = b
            k += n
        null[i] = separability_margin(corr, lab, cols)
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))

    # --- data-driven partition -------------------------------------
    # Average linkage chains on weak structure and produces a giant cluster
    # plus singletons, which inflates nothing but tells us nothing either. We
    # therefore sweep k and both linkages, and flag degenerate solutions
    # explicitly rather than reporting a single ARI as if it were a finding.
    D = 1 - corr.loc[cols, cols].abs().to_numpy()
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    cond = squareform(D, checks=False)
    truth = np.array([BLOCKS.index(block_of(c)) for c in cols])

    # Ward's criterion is only defined for Euclidean distances, and 1-|rho| is
    # not one. Average and complete linkage run on 1-|rho| as before. Ward runs
    # on the features themselves: each feature is a point whose coordinates
    # are its rank-standardised, volume-residualised values across authors, so
    # the distance between two features is Euclidean, sqrt(2(n-1)(1-rho)).
    # Missing values (partially defined features) are set to 0, the column
    # mean, which shrinks those features toward the centre rather than
    # dropping authors.
    linkages = {"average": linkage(cond, method="average"),
                "complete": linkage(cond, method="complete")}
    if X is not None:
        F = X[cols].fillna(0.0).to_numpy(float).T
        linkages["ward"] = linkage(F, method="ward", metric="euclidean")

    sweep = []
    for method, Z in linkages.items():
        for k in range(2, 11):
            lab = fcluster(Z, t=k, criterion="maxclust")
            sizes = np.bincount(lab)[1:]
            degenerate = bool(sizes.max() / len(cols) > 0.80)
            sweep.append({
                "method": method, "k": int(k),
                "ari": float(adjusted_rand_score(truth, lab)),
                "largest_share": float(sizes.max() / len(cols)),
                "n_singletons": int((sizes == 1).sum()),
                "degenerate": degenerate,
            })

    non_deg = [r for r in sweep if not r["degenerate"]]
    best = max(non_deg, key=lambda r: r["ari"]) if non_deg else \
        max(sweep, key=lambda r: r["ari"])
    all_degenerate = len(non_deg) == 0

    Zb = linkages[best["method"]]
    emp = fcluster(Zb, t=best["k"], criterion="maxclust")
    emp_labels = {c: f"E{emp[i]}" for i, c in enumerate(cols)}
    emp_margin = separability_margin(corr, emp_labels, cols)

    comp = {}
    for k in sorted(set(emp)):
        members = [cols[i] for i in range(len(cols)) if emp[i] == k]
        counts = {b: sum(1 for m in members if block_of(m) == b) for b in BLOCKS}
        comp[f"E{k}"] = {"n": len(members), "by_space": counts}

    return {"observed_margin": obs, "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)), "p": p,
            "null": null.tolist(), "ari": best["ari"],
            "best_partition": best, "sweep": sweep,
            "all_degenerate": all_degenerate,
            "empirical_margin": emp_margin, "clusters": comp}


# ======================================================================
# 3. known-groups validity
# ======================================================================

def cv_auc(X: np.ndarray, y: np.ndarray, rng_seed: int = 42,
           folds: int = 5, groups: np.ndarray | None = None
           ) -> Tuple[float, float]:
    """Cross-validated AUC. With groups (matched-pair ids), both members of a
    pair always fall in the same fold, so the model is never tested on the
    partner of an author it was trained on."""
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0))
    if groups is not None and StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True,
                                        random_state=rng_seed)
        splits = splitter.split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True,
                                   random_state=rng_seed)
        splits = splitter.split(X, y)
    aucs = []
    for tr, te in splits:
        model.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], model.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def known_groups(df: pd.DataFrame, X: pd.DataFrame, cols_by_block: Dict,
                 rng) -> Dict:
    out = {}

    # --- contrast A: seed vs control, matched pairs, C/P/S only -------
    sub = df[df.matched == 1] if "matched" in df.columns else df
    idx = sub.index
    y = (sub.cohort == "seed").astype(int).to_numpy()
    groups = sub.pair_id.to_numpy() if "pair_id" in sub.columns else None
    if groups is None:
        print("  WARNING: no pair_id column; rerun extract_cpst_features.py "
              "--rematch. Falling back to ungrouped folds.")
    print(f"  contrast A, seed vs control on {len(sub):,} matched authors")

    # covariate balance on the matching variables, standardised mean difference
    bal = {}
    for name, v in (("log records", np.log1p(sub.n_items.to_numpy(float))),
                    ("log tenure", np.log1p(sub.tenure_days.to_numpy(float)))):
        s1, s0 = v[y == 1], v[y == 0]
        pooled = np.sqrt((s1.var(ddof=1) + s0.var(ddof=1)) / 2)
        bal[name] = {"seed_median": float(np.expm1(np.median(s1))),
                     "control_median": float(np.expm1(np.median(s0))),
                     "smd": float((s1.mean() - s0.mean()) / pooled)}
        print(f"    balance {name:<12} SMD {bal[name]['smd']:+.3f}  "
              f"(medians {bal[name]['seed_median']:.0f} vs "
              f"{bal[name]['control_median']:.0f})")

    # baseline: what volume and tenure alone achieve on the matched sample
    base_X = np.column_stack([np.log1p(sub.n_items.to_numpy(float)),
                              np.log1p(sub.tenure_days.to_numpy(float))])
    auc_base, sd_base = cv_auc(base_X, y, groups=groups)
    print(f"    baseline (log records + log tenure only) AUC {auc_base:.3f} "
          f"(SD {sd_base:.3f})")

    a_rows = []
    combos = [("C", ["C"]), ("P", ["P"]), ("S", ["S"]),
              ("C+P+S", ["C", "P", "S"])]
    for name, bs in combos:
        raw = [c for b in bs for c in cols_by_block[b]]
        adj = [c for c in raw if c not in SELECTION_CONFOUND]
        auc_raw, sd_raw = cv_auc(X.loc[idx, raw].to_numpy(float), y,
                                 groups=groups)
        auc, sd = cv_auc(X.loc[idx, adj].to_numpy(float), y, groups=groups)
        a_rows.append({"features": name, "k": len(adj), "auc": auc, "sd": sd,
                       "k_raw": len(raw), "auc_raw": auc_raw})
        print(f"    {name:<7} k={len(adj):<3} AUC {auc:.3f} (SD {sd:.3f})   "
              f"[unadjusted {auc_raw:.3f} with {len(raw)} features]")

    # thinking space reported only as the circularity demonstration
    cols_t = [c for c in cols_by_block["T"] if c not in SELECTION_CONFOUND]
    auc_t, sd_t = cv_auc(X.loc[idx, cols_t].to_numpy(float), y,
                         groups=groups)
    print(f"    {'T':<7} k={len(cols_t):<3} AUC {auc_t:.3f} "
          f"(SD {sd_t:.3f})  [CIRCULAR, reported as such]")
    out["seed_control"] = {"n": int(len(sub)), "rows": a_rows,
                           "balance": bal,
                           "baseline": {"auc": auc_base, "sd": sd_base},
                           "thinking_circular": {"auc": auc_t, "sd": sd_t,
                                                 "k": len(cols_t)}}

    # --- contrast B: community block, non-circular --------------------
    sub_b = df[df.seed_block.isin(["anxiety", "disengagement"])]
    idx_b = sub_b.index
    yb = (sub_b.seed_block == "disengagement").astype(int).to_numpy()
    print(f"  contrast B, anxiety vs disengagement on {len(sub_b):,} authors "
          f"({yb.sum():,} disengagement)")

    b_rows = []
    for b in BLOCKS:
        cols = [c for c in cols_by_block[b] if c not in COMMUNITY_DEFINING]
        if not cols:
            continue
        auc, sd = cv_auc(X.loc[idx_b, cols].to_numpy(float), yb)
        b_rows.append({"features": BLOCK_NAMES[b], "k": len(cols),
                       "auc": auc, "sd": sd})
        print(f"    {BLOCK_NAMES[b]:<9} k={len(cols):<3} AUC {auc:.3f} "
              f"(SD {sd:.3f})")
    cols_all = [c for b in BLOCKS for c in cols_by_block[b]
                if c not in COMMUNITY_DEFINING]
    auc, sd = cv_auc(X.loc[idx_b, cols_all].to_numpy(float), yb)
    b_rows.append({"features": "All four", "k": len(cols_all),
                   "auc": auc, "sd": sd})
    print(f"    {'All four':<9} k={len(cols_all):<3} AUC {auc:.3f} (SD {sd:.3f})")
    out["community_block"] = {"n": int(len(sub_b)), "rows": b_rows}

    return out


# ======================================================================
# 4. redundancy
# ======================================================================

def whiten(A, tol=1e-8):
    A = A - A.mean(axis=0)
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    if s[0] <= 0:
        return U[:, :0]
    return U[:, :int((s > tol * s[0]).sum())]


def pcs_to(A, frac=0.8):
    A = A - A.mean(axis=0)
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    ev = s ** 2 / np.sum(s ** 2)
    k = int(np.searchsorted(np.cumsum(ev), frac) + 1)
    return U[:, :k] * s[:k]


def redundancy(X: pd.DataFrame, cols_by_block: Dict) -> List[Dict]:
    rows = []
    for a, b in combinations(BLOCKS, 2):
        sub = X[cols_by_block[a] + cols_by_block[b]].dropna()
        Pa = pcs_to(sub[cols_by_block[a]].to_numpy())
        Pb = pcs_to(sub[cols_by_block[b]].to_numpy())
        Ua, Ub = whiten(Pa), whiten(Pb)
        if Ua.shape[1] == 0 or Ub.shape[1] == 0:
            continue
        U, s, Vt = np.linalg.svd(Ua.T @ Ub)
        u = Ua @ U[:, 0]
        v = Ub @ Vt[0, :]
        # mean squared cross-loading: variance of one block explained by the
        # other block's first canonical variate
        rb_given_a = float(np.mean([
            stats.spearmanr(u, sub[c].to_numpy(float)).statistic ** 2
            for c in cols_by_block[b]]))
        ra_given_b = float(np.mean([
            stats.spearmanr(v, sub[c].to_numpy(float)).statistic ** 2
            for c in cols_by_block[a]]))
        rows.append({"a": a, "b": b, "rho1": float(np.clip(s[0], 0, 1)),
                     "red_b_given_a": rb_given_a, "red_a_given_b": ra_given_b})
        print(f"  {a}-{b}  rho1 {s[0]:.3f}  "
              f"var({b}|{a}) {rb_given_a:.3f}  var({a}|{b}) {ra_given_b:.3f}")
    return rows


# ======================================================================
# figures
# ======================================================================

def fig9_partition(pt: Dict):
    fig, axes = plt.subplots(1, 2, figsize=(COL_2, 2.2),
                             gridspec_kw={"width_ratios": [1.3, 1]})

    ax = axes[0]
    null = np.array(pt["null"])
    ax.hist(null, bins=40, color=INK_MUTED, alpha=0.65, lw=0, density=True)
    ax.axvline(pt["observed_margin"], color="#2a78d6", lw=2)
    ax.text(pt["observed_margin"], ax.get_ylim()[1] * 0.92, "  CPST",
            color="#2a78d6", fontsize=7.5, va="top")
    ax.axvline(pt["empirical_margin"], color="#eb6834", lw=2, ls="--")
    ax.text(pt["empirical_margin"], ax.get_ylim()[1] * 0.72, "  data-driven",
            color="#eb6834", fontsize=7.5, va="top")
    ax.set_xlabel("separability margin, median $|\\rho|$")
    ax.set_ylabel("density")
    ax.set_title("Against random partitions of the same sizes", loc="left")
    ax.grid(axis="y", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    clusters = pt["clusters"]
    names = list(clusters.keys())
    bottom = np.zeros(len(names))
    for b in BLOCKS:
        vals = np.array([clusters[n]["by_space"][b] for n in names], float)
        ax.bar(np.arange(len(names)), vals, bottom=bottom, width=0.62,
               color=BLOCK_COLOR[b], lw=0, label=BLOCK_NAMES[b])
        bottom += vals
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_xlabel("empirical cluster")
    ax.set_ylabel("features")
    ax.set_title(f"Composition, ARI = {pt['ari']:.2f}", loc="left")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(axis="y", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    save(fig, "fig9_partition_validity")


def fig10_reliability(rel: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(COL_1, 2.4))
    for i, b in enumerate(BLOCKS):
        vals = rel[rel.block == b].r_sb.dropna().to_numpy()
        if not len(vals):
            continue
        xs = np.random.default_rng(i).normal(i, 0.055, len(vals))
        ax.scatter(xs, vals, s=14, color=BLOCK_COLOR[b], alpha=0.7, lw=0)
        ax.plot([i - 0.22, i + 0.22], [np.median(vals)] * 2,
                color=BLOCK_COLOR[b], lw=2.5, solid_capstyle="butt")
    ax.axhline(0.70, color=INK_2, lw=1, ls=":")
    ax.text(3.45, 0.715, "0.70", ha="right", fontsize=6.5, color=INK_2)
    ax.set_xticks(range(4))
    ax.set_xticklabels([BLOCK_NAMES[b] for b in BLOCKS])
    ax.set_ylabel("split-half reliability (Spearman-Brown)")
    ax.set_ylim(-0.05, 1.02)
    ax.grid(axis="y", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Feature reliability by space", loc="left")
    save(fig, "fig10_reliability")


# ======================================================================
# LaTeX
# ======================================================================

def write_tables(rel: pd.DataFrame, pt: Dict, kg: Dict, red: List[Dict]):
    # Table V: reliability by block
    body = []
    for b in BLOCKS:
        v = rel[rel.block == b].r_sb.dropna()
        if not len(v):
            continue
        body.append(f"{BLOCK_NAMES[b]} & {len(v)} & {v.median():.2f} & "
                    f"{v.min():.2f} & {v.max():.2f} & "
                    f"{(v >= 0.7).sum()} \\\\")
    tex = r"""\begin{table}[!t]
\caption{Split-half reliability by space, Spearman-Brown corrected. Halves are
alternating records per author, so each spans the full observation window.}
\label{tab:reliability}
\centering
\begin{tabular}{lrrrrr}
\toprule
Space & $k$ & Median & Min & Max & $\geq 0.70$ \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(os.path.join(OUT_DIR, "table5_reliability.tex"), "w",
         encoding="utf-8").write(tex)

    # Table VI: known groups
    a = kg["seed_control"]
    b = kg["community_block"]
    arows = "\n".join(f"{r['features']} & {r['k']} & {r['auc']:.3f} & "
                      f"{r['sd']:.3f} \\\\" for r in a["rows"])
    arows += (f"\n\\textit{{Volume + tenure only}} & 2 & "
              f"\\textit{{{a['baseline']['auc']:.3f}}} & "
              f"\\textit{{{a['baseline']['sd']:.3f}}} \\\\")
    print("  contrast A, with vs without design-encoding features:")
    for r in a["rows"]:
        print(f"    {r['features']:<7} {r['auc_raw']:.3f} (k={r['k_raw']}) -> "
              f"{r['auc']:.3f} (k={r['k']})")
    brows = "\n".join(f"{r['features']} & {r['k']} & {r['auc']:.3f} & "
                      f"{r['sd']:.3f} \\\\" for r in b["rows"])
    tex = r"""\begin{table}[!t]
\caption{Known-groups validity. Five-fold cross-validated AUC. Contrast A
excludes the thinking space because seed authors were selected on their text;
the circular value is reported beneath for reference. Contrast B excludes the
two community-share features, which define the groups.}
\label{tab:knowngroups}
\centering
\begin{tabular}{lrrr}
\toprule
Features & $k$ & AUC & SD \\
\midrule
\multicolumn{4}{l}{\textit{A. Seed vs control, """ + f"{a['n']}" + \
r""" matched authors}} \\
""" + arows + r"""
\midrule
\multicolumn{4}{l}{\textit{B. Anxiety vs disengagement, """ + f"{b['n']}" + \
r""" authors}} \\
""" + brows + r"""
\bottomrule
\end{tabular}

\vspace{1mm}
\footnotesize
Thinking space on contrast A reaches AUC """ + \
f"{a['thinking_circular']['auc']:.3f}" + r""", which is circular by
construction and is not evidence of validity.
\end{table}
"""
    open(os.path.join(OUT_DIR, "table6_knowngroups.tex"), "w",
         encoding="utf-8").write(tex)

    # Table VII: redundancy
    rows = "\n".join(
        f"{BLOCK_NAMES[r['a']]}--{BLOCK_NAMES[r['b']]} & {r['rho1']:.3f} & "
        f"{r['red_b_given_a']:.3f} & {r['red_a_given_b']:.3f} \\\\" for r in red)
    tex = r"""\begin{table}[!t]
\caption{Canonical correlation and redundancy. Redundancy is the mean squared
cross-loading: the share of one block's variance aligned with the other block's
first canonical variate.}
\label{tab:redundancy}
\centering
\begin{tabular}{lrrr}
\toprule
Pair $A$--$B$ & $\rho_1$ & var($B|A$) & var($A|B$) \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(os.path.join(OUT_DIR, "table7_redundancy.tex"), "w",
         encoding="utf-8").write(tex)
    print("  wrote table5_reliability.tex, table6_knowngroups.tex, "
          "table7_redundancy.tex")


# ======================================================================

def _args(ap):
    """Parsed arguments, safe under Jupyter.

    In a notebook sys.argv carries the kernel's own flags, and argparse's
    prefix matching binds --f=<kernel>.json to --features without complaint,
    so the script runs against the kernel connection file. Under ipykernel we
    therefore ignore sys.argv entirely and use the defaults; pass options from
    a terminal instead, or edit the constants at the top of the file.
    """
    in_notebook = ("ipykernel" in sys.modules) or any(
        ("ipykernel_launcher" in a) or ("kernel-" in a and a.endswith(".json"))
        for a in sys.argv)
    return ap.parse_args([] if in_notebook else sys.argv[1:])


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--features", default=os.path.join(BASE_DIR, "cpst_features.csv"))
    ap.add_argument("--h1", default=os.path.join(BASE_DIR, "cpst_features_h1.csv"))
    ap.add_argument("--h2", default=os.path.join(BASE_DIR, "cpst_features_h2.csv"))
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = _args(ap)

    os.makedirs(OUT_DIR, exist_ok=True)
    setup_style()
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.features)
    feat_cols = [c for c in df.columns
                 if c not in NON_FEATURES and block_of(c) in BLOCKS
                 and c not in DROP_FEATURES]
    cols_by_block = {b: [c for c in feat_cols if block_of(c) == b] for b in BLOCKS}
    print(f"author-windows: {len(df):,}, features: {len(feat_cols)}")

    covars = pd.DataFrame({"log_n": np.log1p(df.n_items.to_numpy(float)),
                           "log_t": np.log1p(df.tenure_days.to_numpy(float))})
    X = rank_z(residualise(df[feat_cols], covars))
    corr = X.corr(method="spearman")

    results = {}

    # ---- 1 ----
    print("\n1. split-half reliability")
    rel = pd.DataFrame()
    if os.path.exists(args.h1) and os.path.exists(args.h2):
        h1, h2 = pd.read_csv(args.h1), pd.read_csv(args.h2)
        rel = reliability(h1, h2, feat_cols)
        for b in BLOCKS:
            v = rel[rel.block == b].r_sb.dropna()
            if len(v):
                print(f"  {BLOCK_NAMES[b]:<9} median {v.median():.3f}  "
                      f"range {v.min():.3f}-{v.max():.3f}  "
                      f"{(v>=0.7).sum()}/{len(v)} at or above 0.70")
        rel.to_csv(os.path.join(OUT_DIR, "reliability.csv"), index=False)
        results["reliability"] = rel.to_dict("records")
    else:
        print("  half-sample files not found. Run:")
        print("    python extract_cpst_features.py --half 1")
        print("    python extract_cpst_features.py --half 2")

    dis = {}
    if len(rel):
        dis = disattenuated_separability(corr, feat_cols, rel)
        print("\n1b. separability corrected for unreliability")
        print(f"  {'space':<10}{'rel':>7}{'within':>9}{'within*':>9}"
              f"{'betw':>8}{'betw*':>8}{'gap*':>8}")
        for b in BLOCKS:
            d = dis[b]
            print(f"  {BLOCK_NAMES[b]:<10}{d['rel_median']:>7.3f}"
                  f"{d['within_obs']:>9.3f}{d['within_dis']:>9.3f}"
                  f"{d['between_obs']:>8.3f}{d['between_dis']:>8.3f}"
                  f"{d['gap_dis']:>+8.3f}")
        print("  * = disattenuated")
        results["disattenuated"] = dis

    # ---- 2 ----
    print("\n2. is the CPST partition better than an arbitrary one?")
    pt = partition_tests(corr, feat_cols, args.n_perm, rng, X=X)
    print(f"  observed margin {pt['observed_margin']:+.4f}")
    print(f"  random-partition null: mean {pt['null_mean']:+.4f}, "
          f"p95 {pt['null_p95']:+.4f}, p = {pt['p']:.4f}")
    bp = pt["best_partition"]
    if pt["all_degenerate"]:
        print("  every clustering solution is degenerate (one cluster holds")
        print("  more than 80% of features). The correlation structure is too")
        print("  weak to support a data-driven partition at all.")
    print(f"  best non-degenerate: {bp['method']} linkage, k={bp['k']}, "
          f"ARI = {bp['ari']:.3f}, largest cluster "
          f"{100*bp['largest_share']:.0f}% of features")
    print(f"  its separability margin {pt['empirical_margin']:+.4f}")
    for k, v in pt["clusters"].items():
        comp = ", ".join(f"{BLOCK_NAMES[b]} {n}" for b, n in v["by_space"].items() if n)
        print(f"    {k} (n={v['n']}): {comp}")
    results["partition"] = {k: v for k, v in pt.items() if k != "null"}

    # ---- 3 ----
    print("\n3. known-groups validity")
    kg = known_groups(df, X, cols_by_block, rng)
    results["known_groups"] = kg

    # ---- 4 ----
    print("\n4. redundancy")
    red = redundancy(X, cols_by_block)
    results["redundancy"] = red

    # ---- outputs ----
    print("\nfigures")
    fig9_partition(pt)
    if len(rel):
        fig10_reliability(rel)

    print("tables")
    if len(rel):
        write_tables(rel, pt, kg, red)
    else:
        print("  skipping reliability table, half-sample files missing")

    with open(os.path.join(OUT_DIR, "results_validity.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # ---- verdict ----
    print("\n" + "=" * 66)
    print("READING THE RESULT")
    print("=" * 66)
    if pt["p"] < 0.05:
        print(f"The CPST partition beats random partitions of the same sizes "
              f"(p = {pt['p']:.4f}).")
        print("The assignment of features to spaces carries information.")
    else:
        print(f"The CPST partition does NOT beat random partitions "
              f"(p = {pt['p']:.4f}).")
        print("Any grouping of these features would separate this well. This is")
        print("a negative result about the partition and must be reported.")
    if pt["all_degenerate"]:
        print("No data-driven partition is recoverable: every clustering")
        print("solution is degenerate. Report this rather than an ARI, and")
        print("note that it is consistent with the small separability margin.")
    elif pt["ari"] > 0.4:
        print(f"The data's own grouping largely recovers CPST (ARI = {pt['ari']:.2f}).")
    elif pt["ari"] > 0.15:
        print(f"The data's grouping partly overlaps CPST (ARI = {pt['ari']:.2f}).")
    else:
        print(f"The data organise themselves differently from CPST "
              f"(ARI = {pt['ari']:.2f}).")
        print("Report the empirical cluster composition; it says which spaces")
        print("the data merges or splits.")

    a = kg["seed_control"]
    cps = [r for r in a["rows"] if r["features"] == "C+P+S"][0]
    print(f"\nSeed vs control on Cyber+Physical+Social alone: "
          f"AUC {cps['auc']:.3f}")
    if cps["auc"] > 0.60:
        print("Non-textual spaces discriminate the cohorts. Known-groups")
        print("validity is supported without circularity.")
    elif cps["auc"] > 0.55:
        print("Weak but above chance. Report the magnitude honestly.")
    else:
        print("At or near chance. The non-textual spaces do not distinguish")
        print("keyword-seeded authors, which bounds what they can be said to")
        print("measure about this construct.")

    if len(rel):
        weak = [BLOCK_NAMES[b] for b in BLOCKS
                if len(rel[rel.block == b].r_sb.dropna())
                and rel[rel.block == b].r_sb.median() < 0.6]
        if weak:
            print(f"\nLow-reliability space(s): {', '.join(weak)}. A block of")
            print("unreliable features cannot cohere whatever it measures, so")
            print("check this before attributing non-coherence to the construct.")
    print("=" * 66)
    print(f"\noutputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
