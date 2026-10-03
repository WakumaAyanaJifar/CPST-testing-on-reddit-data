#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper 1 analysis, version 2: corrected separability and a redesigned validity test
==================================================================================

Version 1 produced canonical correlations of 0.93 to 1.00 for every block pair.
Those were artifacts, not findings. This version fixes three things and replaces
the convergence test with a design that can actually detect the effect.

WHAT WENT WRONG IN V1

  1. Algebraic dependence between blocks.
     C_post_share + S_comment_share = 1 exactly, so the C-S canonical correlation
     was 1.000 by construction. Several features were also deterministic functions
     of two others within their own block (T_affect_balance, T_temporal_balance,
     T_self_other_ratio) and one compositional triple summed to 1
     (C_anxiety_share + C_diseng_share + C_offblock_share).
     FIX: drop the redundant member of every exact dependency (DROP_FEATURES),
     then verify numerically that no feature is a linear combination of the rest.

  2. Activity volume as a shared driver.
     How much a person posts moves features in all four blocks at once. Left in,
     it inflates between-block correlation and deflates apparent separability.
     FIX: residualise every feature on log record count and log tenure before
     the separability analysis. Separability is then a claim about structure
     that survives volume, which is the claim worth making.

  3. Saturated canonical subspaces.
     With 15 to 27 features per block and exact dependencies present, CCA fits a
     near-saturated subspace.
     FIX: truncate each block to the principal components covering 80% of its
     variance before CCA, and keep the permutation null.

WHAT THE VALIDITY TEST BECOMES

  V1 correlated two author-level aggregates: share of activity in the local night
  against sleep-term rate across all of an author's text. Both are diluted by the
  overwhelming majority of records that concern neither. An author-level
  correlation of 0.05 is what dilution looks like, and it cannot distinguish a
  weak effect from a well-measured null.

  V2 tests the same idea at record level, within author:

      for each author, P(sleep term | posted in local night)
                   vs P(sleep term | posted in local day)

  paired across authors. Every between-author confound (language, volume,
  community, geography) is removed by the pairing, which is precisely what
  defeated the v1 negative control.

  Note what this validates. It validates THE OFFSET ESTIMATOR, by showing that
  the hour window it calls "local night" is when people write about not sleeping.
  That is the precondition for every local-time feature, and it should be
  reported as estimator validation rather than as validation of the construct.

  A second, independent estimator check needs no text at all: if the offsets are
  real they should pile up at inhabited timezones rather than spreading uniformly.

Usage
    python analyze_spaces_v2.py
    python analyze_spaces_v2.py --sample matched
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Data folder: set CPST_BASE_DIR to override the default location.
BASE_DIR = os.environ.get(
    "CPST_BASE_DIR",
    os.path.join(os.path.expanduser("~"), "Documents", "CPST_FOMO_v6"))
FEATURES = os.path.join(BASE_DIR, "cpst_features.csv")
DB_PATH = os.path.join(BASE_DIR, "cpst_fomo.db")
OUT_DIR = os.path.join(BASE_DIR, "paper1")

BLOCKS = ["C", "P", "S", "T"]
BLOCK_NAMES = {"C": "Cyber", "P": "Physical", "S": "Social", "T": "Thinking"}
BLOCK_COLOR = {"C": "#2a78d6", "P": "#eb6834", "S": "#1baf7a", "T": "#4a3aa7"}
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8985"
GRID, SURFACE = "#e6e5e1", "#ffffff"
DIVERGING = LinearSegmentedColormap.from_list(
    "cpst_div", ["#0d366b", "#256abf", "#86b6ef", "#f0efec",
                 "#f0a09f", "#cf2a29", "#7d1413"])
COL_1, COL_2 = 3.5, 7.16

WINDOW_START_TS = datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp()
WINDOW_END_TS = datetime(2025, 12, 31, tzinfo=timezone.utc).timestamp()

NON_FEATURES = {"pair_id", "author_hash", "cohort", "seed_block", "n_items", "tenure_days",
                "first_seen", "last_seen", "matched"}

# Exact dependencies and nuisance parameters removed.
# P_utc_offset_est is the LOCATION PARAMETER used to derive the local-time features, not a measurement of anyone's physical state. Left in the feature matrix it injects geography, and therefore language background, into the Physical block.
DROP_FEATURES = {
    "S_comment_share":    "= 1 - C_post_share",
    "C_offblock_share":   "= 1 - C_anxiety_share - C_diseng_share",
    "T_affect_balance":   "function of T_affect_pos and T_affect_neg",
    "T_temporal_balance": "function of T_past and T_future",
    "T_self_other_ratio": "function of T_self_sg and T_other_ref",
    "S_max_depth":        "near-duplicate of S_mean_depth on the same subset",
    "P_utc_offset_est":   "location parameter, not a physical-state measure",
}

# Features that are raw counts and therefore scale with exposure. Linear
# residualisation on log record count does not remove that structure, so the
# P-S coupling in the first run was driven by them. --drop-counts removes them
# and keeps only rates, proportions and shape measures, which is the robustness
# run that decides whether a coupling is about tempo or about amount.
COUNT_SCALING = {
    "C_n_communities", "C_n_link_domains",
    "P_n_sessions", "P_active_days",
    "S_replies_received", "S_n_partners", "S_mutual_partners", "S_n_threads",
    "T_tokens_total",
}

# Features defined only on the thread-completed subset. Because CCA drops
# incomplete rows, any block pair containing one of these is estimated on
# roughly half the authors. --drop-partial removes them so the same analysis
# can be run on the full sample, which is the check that decides whether a
# coupling depends on a partially observed feature.
PARTIAL_COVERAGE_FEATURES = {
    "P_reply_latency_med", "P_fast_reply_share", "S_mean_depth",
}

# Lexicons must match the extractor exactly, or the record-level test is not
# measuring the same thing as the author-level feature.
SLEEP_RE = re.compile(
    r"\b(sleep|asleep|insomnia|awake|3am|4am|2am|all night|couldn't sleep|"
    r"stayed up|tired)\b", re.I)
SOMATIC_RE = re.compile(
    r"\b(heart racing|chest|breathe|breathing|nausea|nauseous|"
    r"sick to my stomach|shaking|dizzy|exhausted|headache)\b", re.I)
COMPULSION_RE = re.compile(
    r"\b(keep checking|can't stop|refresh|scrolling|again and again|"
    r"every few minutes|compulsive|constantly checking)\b", re.I)
# Sleep terms without clock-time tokens ("2am", "3am", "4am"), which people
# write when stating the current time and which would make the test partly
# a check of the timestamp against itself.
SLEEP_NOCLOCK_RE = re.compile(
    r"\b(sleep|asleep|insomnia|awake|all night|couldn't sleep|"
    r"stayed up|tired)\b", re.I)
# Controls: no plausible diurnal link, similar prevalence.
QUESTION_RE = re.compile(r"\?")
SECOND_RE = re.compile(r"\b(you|your|yours|yourself)\b", re.I)

PROBES = [
    ("sleep",      SLEEP_RE,      "test",    "sleep terms"),
    ("sleep_noclock", SLEEP_NOCLOCK_RE, "test", "sleep, no clock"),
    ("somatic",    SOMATIC_RE,    "test",    "somatic terms"),
    ("compulsion", COMPULSION_RE, "test",    "compulsion terms"),
    ("question",   QUESTION_RE,   "control", "question marks"),
    ("second",     SECOND_RE,     "control", "second person"),
]

NIGHT_LO, NIGHT_HI = 0, 5        # local hours counted as night
DAY_LO, DAY_HI = 9, 22           # local hours counted as day
MIN_PER_CELL = 10                # records needed in each cell per author


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.size": 8,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "axes.edgecolor": INK_2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.labelcolor": INK,
        "grid.color": GRID, "grid.linewidth": 0.5,
        "lines.linewidth": 2.0, "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


TAG = ""


def save(fig, name):
    name = f"{name}{TAG}"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"),
                    dpi=600 if ext == "png" else None)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def block_of(c): return c.split("_", 1)[0]


def rank_z(df):
    r = df.rank(method="average")
    return (r - r.mean()) / r.std(ddof=1)


def whiten(A, tol=1e-8):
    A = A - A.mean(axis=0)
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    if s[0] <= 0:
        return U[:, :0]
    return U[:, :int((s > tol * s[0]).sum())]


def cca_first(X, Y):
    Ux, Uy = whiten(X), whiten(Y)
    if Ux.shape[1] == 0 or Uy.shape[1] == 0:
        return 0.0
    return float(np.clip(np.linalg.svd(Ux.T @ Uy, compute_uv=False)[0], 0, 1))


def cca_variates(X, Y):
    """First canonical correlation plus the pair of canonical variates.

    The variates are what make a canonical correlation interpretable. Structure
    coefficients, the correlation of each original feature with its own side's
    variate, say WHAT the two blocks share. A coupling driven entirely by
    activity-like features is a different claim from one driven by content.
    """
    Ux, Uy = whiten(X), whiten(Y)
    if Ux.shape[1] == 0 or Uy.shape[1] == 0:
        return 0.0, None, None
    U, s, Vt = np.linalg.svd(Ux.T @ Uy)
    u = Ux @ U[:, 0]
    v = Uy @ Vt[0, :]
    return float(np.clip(s[0], 0, 1)), u, v


def structure_coefficients(variate: np.ndarray, raw: pd.DataFrame,
                           top: int = 5) -> List[Tuple[str, float]]:
    out = []
    for col in raw.columns:
        y = raw[col].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(variate)
        if ok.sum() < 50:
            continue
        r = stats.spearmanr(variate[ok], y[ok]).statistic
        if np.isfinite(r):
            out.append((col, float(r)))
    out.sort(key=lambda t: -abs(t[1]))
    return out[:top]


def mantel_haenszel(tables: np.ndarray) -> Dict:
    """Common odds ratio across strata, with a Robins-Breslow-Greenland CI.

    Each row of `tables` is one author's 2x2: [night_hit, night_miss,
    day_hit, day_miss]. Stratifying by author removes every between-author
    confound while using the records directly, so it does not lose power to
    the ties that defeat a paired test on sparse proportions.
    """
    a, b, c, d = tables[:, 0], tables[:, 1], tables[:, 2], tables[:, 3]
    n = a + b + c + d
    keep = n > 0
    a, b, c, d, n = a[keep], b[keep], c[keep], d[keep], n[keep]

    R = a * d / n
    S = b * c / n
    sR, sS = R.sum(), S.sum()
    if sR <= 0 or sS <= 0:
        return {"or": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan,
                "n_strata": int(len(a)), "n_records": int(n.sum())}
    orr = sR / sS

    P = (a + d) / n
    Q = (b + c) / n
    var = (np.sum(P * R) / (2 * sR ** 2)
           + np.sum(P * S + Q * R) / (2 * sR * sS)
           + np.sum(Q * S) / (2 * sS ** 2))
    se = np.sqrt(var)
    lo, hi = np.exp(np.log(orr) - 1.96 * se), np.exp(np.log(orr) + 1.96 * se)

    E = (a + b) * (a + c) / n
    V = ((a + b) * (c + d) * (a + c) * (b + d)) / (n ** 2 * (n - 1))
    V = np.where(np.isfinite(V), V, 0.0)
    if V.sum() <= 0:
        p = np.nan
    else:
        chi = (abs(a.sum() - E.sum()) - 0.5) ** 2 / V.sum()
        p = float(stats.chi2.sf(chi, 1))
    return {"or": float(orr), "lo": float(lo), "hi": float(hi), "p": p,
            "n_strata": int(len(a)), "n_records": int(n.sum())}


def pcs_to(A, frac=0.8):
    """Principal component scores covering `frac` of a block's variance."""
    A = A - A.mean(axis=0)
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    ev = s ** 2 / np.sum(s ** 2)
    k = int(np.searchsorted(np.cumsum(ev), frac) + 1)
    return U[:, :k] * s[:k], ev


def residualise(df: pd.DataFrame, covars: pd.DataFrame) -> pd.DataFrame:
    """Remove the linear contribution of the covariates from every feature.

    Volume is the obvious shared driver across all four blocks; if separability
    only holds because active people score high everywhere, it is not
    separability of the spaces.
    """
    Z = np.column_stack([np.ones(len(covars))] + [covars[c].to_numpy()
                                                  for c in covars.columns])
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


def check_collinearity(X: pd.DataFrame, thresh: float = 0.999) -> List[str]:
    """Flag any feature that is essentially a linear combination of the others."""
    flagged = []
    A = X.dropna()
    cols = list(A.columns)
    M = A.to_numpy()
    M = (M - M.mean(0)) / np.where(M.std(0) > 0, M.std(0), 1)
    for i, c in enumerate(cols):
        y = M[:, i]
        Z = np.delete(M, i, axis=1)
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        r2 = 1 - np.sum((y - Z @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
        if r2 > thresh:
            flagged.append(f"{c} (R2={r2:.4f})")
    return flagged


def holm(p):
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj.tolist()


# ======================================================================
# offset estimator validation
# ======================================================================

def offset_distribution(df: pd.DataFrame) -> Dict:
    """If estimated offsets are real they concentrate at inhabited timezones.
    A uniform spread would mean the estimator is reading noise."""
    off = df.P_utc_offset_est.dropna().astype(int).to_numpy()
    counts = np.array([np.sum(off == h) for h in range(-12, 13)])
    exp = np.full(25, counts.sum() / 25)
    chi2, p = stats.chisquare(counts, exp)
    # concentration in the Americas + western Europe band, where an
    # English-language platform's users overwhelmingly sit
    band = np.sum(counts[np.array([h in range(-8, 3) for h in range(-12, 13)])])
    return {"counts": counts.tolist(), "hours": list(range(-12, 13)),
            "chi2": float(chi2), "p": float(p),
            "band_share": float(band / max(counts.sum(), 1))}


def record_level_convergence(db: str, offsets: Dict[str, int],
                             rng) -> List[Dict]:
    """Within-author night vs day probability for each lexical probe."""
    counts = defaultdict(lambda: defaultdict(int))   # author -> key -> count

    conn = sqlite3.connect(db)
    cur = conn.execute("""
        SELECT author_hash, created_utc, body_raw
        FROM items
        WHERE author_hash IS NOT NULL
          AND body_raw IS NOT NULL
          AND created_utc BETWEEN ? AND ?
    """, (WINDOW_START_TS, WINDOW_END_TS))

    seen = 0
    while True:
        chunk = cur.fetchmany(20000)
        if not chunk:
            break
        for author, ts, text in chunk:
            off = offsets.get(author)
            if off is None or not text:
                continue
            hour = (int(ts // 3600) + off) % 24
            if NIGHT_LO <= hour < NIGHT_HI:
                cell = "night"
            elif DAY_LO <= hour < DAY_HI:
                cell = "day"
            else:
                continue
            c = counts[author]
            c[f"n_{cell}"] += 1
            for name, rx, _kind, _label in PROBES:
                if rx.search(text):
                    c[f"{name}_{cell}"] += 1
        seen += len(chunk)
        if seen % 200000 == 0:
            print(f"    record pass: {seen:,}")
    conn.close()

    eligible = [a for a, c in counts.items()
                if c["n_night"] >= MIN_PER_CELL and c["n_day"] >= MIN_PER_CELL]
    print(f"  authors with >={MIN_PER_CELL} records in both cells: {len(eligible):,}")

    rows = []
    for name, _rx, kind, label in PROBES:
        pn, pd_, tables = [], [], []
        for a in eligible:
            c = counts[a]
            nn, nd = c["n_night"], c["n_day"]
            hn, hd = c[f"{name}_night"], c[f"{name}_day"]
            pn.append(hn / nn)
            pd_.append(hd / nd)
            tables.append([hn, nn - hn, hd, nd - hd])
        pn, pd_ = np.array(pn), np.array(pd_)
        mh = mantel_haenszel(np.array(tables, dtype=float))

        # Author-level paired test is kept as a secondary, conservative check.
        # It loses power to ties when per-author hit counts are near zero, which
        # is why the Mantel-Haenszel estimate is the primary result.
        try:
            _stat, p_wil = stats.wilcoxon(pn, pd_, alternative="two-sided",
                                          zero_method="zsplit")
        except ValueError:
            p_wil = np.nan

        rows.append({
            "probe": name, "label": label, "kind": kind,
            "n_authors": len(eligible), "n_records": mh["n_records"],
            "p_night": float(np.mean(pn)), "p_day": float(np.mean(pd_)),
            "ratio": float(np.mean(pn) / max(np.mean(pd_), 1e-9)),
            "or": mh["or"], "or_lo": mh["lo"], "or_hi": mh["hi"],
            "p": mh["p"], "p_wilcoxon": float(p_wil) if p_wil == p_wil else None,
        })
        print(f"  {label:<18} night {np.mean(pn):.4f}  day {np.mean(pd_):.4f}  "
              f"OR {mh['or']:.3f} [{mh['lo']:.3f}, {mh['hi']:.3f}]  ({kind})")

    for r, pa in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = pa
    return rows


# ======================================================================
# figures
# ======================================================================

def fig3_offsets(od: Dict):
    fig, ax = plt.subplots(figsize=(COL_1, 1.9))
    hours, counts = od["hours"], np.array(od["counts"])
    ax.bar(hours, counts, width=0.8, color="#eb6834", lw=0)
    ax.axhline(counts.sum() / len(counts), color=INK_2, lw=1.2, ls="--")
    ax.text(12, counts.sum() / len(counts) * 1.12, "uniform", ha="right",
            fontsize=6.5, color=INK_2)
    ax.set_xlabel("estimated UTC offset (h)")
    ax.set_ylabel("authors")
    ax.set_xticks(range(-12, 13, 4))
    ax.grid(axis="y", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Estimated circadian offsets", loc="left")
    save(fig, "fig3_offset_distribution")


def fig7b_record_level(rows: List[Dict]):
    fig, ax = plt.subplots(figsize=(COL_1, 2.2))
    y = np.arange(len(rows))
    h = 0.34
    for i, r in enumerate(rows):
        col = "#eb6834" if r["kind"] == "test" else INK_MUTED
        ax.barh(i - h / 2 - 0.02, r["p_night"], height=h, color=col, lw=0)
        ax.barh(i + h / 2 + 0.02, r["p_day"], height=h, color=col,
                alpha=0.35, lw=0)
        mark = "*" if r["p_holm"] < 0.05 else ""
        ax.text(max(r["p_night"], r["p_day"]) * 1.04, i,
                f"x{r['ratio']:.2f}{mark}", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("P(term present in a record)")
    ax.grid(axis="x", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[
        Line2D([], [], color="#eb6834", lw=6, label="local night (00–05 h)"),
        Line2D([], [], color="#eb6834", lw=6, alpha=0.35,
               label="local day (09–22 h)")], loc="upper right")
    ax.set_title("Within-author night vs day", loc="left")
    save(fig, "fig7b_record_level_convergence")


def fig5b_within_between(within, between, per_block):
    fig = plt.figure(figsize=(COL_2, 2.4))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.5, 1, 1, 1, 1], wspace=0.35)
    bins = np.linspace(0, 1, 41)

    def tick(a, v, colour):
        a.plot([v, v], [1.0, 1.05], transform=a.get_xaxis_transform(),
               color=colour, lw=1.8, clip_on=False, solid_capstyle="butt")

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(within, bins=bins, density=True, color="#2a78d6", alpha=0.75, lw=0,
            label="within block")
    ax.hist(between, bins=bins, density=True, color=INK_MUTED, alpha=0.6, lw=0,
            label="between blocks")
    tick(ax, np.median(between), INK_2)
    tick(ax, np.median(within), "#184f95")
    ax.set_xlabel("|Spearman $\\rho$|, volume removed")
    ax.set_ylabel("density")
    ax.set_title("All features", loc="left", pad=12)
    ax.legend(loc="upper right")
    ax.grid(axis="y", lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    for i, b in enumerate(BLOCKS):
        a = fig.add_subplot(gs[0, i + 1])
        w, o = per_block[b]
        a.hist(w, bins=bins, density=True, color=BLOCK_COLOR[b], alpha=0.75, lw=0)
        a.hist(o, bins=bins, density=True, color=INK_MUTED, alpha=0.55, lw=0)
        tick(a, np.median(o), INK_2)
        tick(a, np.median(w), BLOCK_COLOR[b])
        a.set_title(BLOCK_NAMES[b], loc="left", color=BLOCK_COLOR[b], pad=12)
        a.set_xlabel("|$\\rho$|")
        a.set_yticks([])
        a.set_xlim(0, 1)
        a.grid(axis="y", lw=0.5)
        a.set_axisbelow(True)
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
    save(fig, "fig5b_within_between_residualised")


def fig6b_canonical(rows):
    fig, ax = plt.subplots(figsize=(COL_1, 2.4))
    y = np.arange(len(rows))
    ax.barh(y, [r["rho1"] for r in rows], height=0.55, color="#2a78d6", lw=0)
    ax.barh(y, [r["null_mean"] for r in rows], height=0.55, color="none",
            edgecolor=INK_2, lw=1.2, linestyle="--")
    for i, r in enumerate(rows):
        mark = "*" if r["p_holm"] < 0.05 else ""
        ax.text(max(r["rho1"], r["null_mean"]) + 0.015, i,
                f"{r['rho1']:.2f}{mark}", va="center", fontsize=7, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['a']}–{r['b']}" for r in rows])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("first canonical correlation (PC-truncated)")
    ax.invert_yaxis()
    ax.grid(axis="x", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[
        Line2D([], [], color="#2a78d6", lw=6, label="observed"),
        Line2D([], [], color=INK_2, lw=1.2, ls="--", label="permutation null")],
        loc="upper right", bbox_to_anchor=(1.0, 0.98))
    ax.set_title("Block-pair canonical correlation", loc="left")
    save(fig, "fig6b_canonical_corrected")


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
    ap.add_argument("--features", default=FEATURES)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--sample", choices=["all", "matched"], default="all")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--offset-trust", type=float, default=0.75)
    ap.add_argument("--skip-records", action="store_true",
                    help="skip the record-level pass over the database")
    ap.add_argument("--drop-counts", action="store_true",
                    help="robustness run: remove exposure-scaling count features")
    ap.add_argument("--drop-partial", action="store_true",
                    help="remove features defined only on the thread-completed subset")
    ap.add_argument("--tag", default="",
                    help="suffix for output filenames, e.g. _nocounts")
    ap.add_argument("--seed", type=int, default=42)
    args = _args(ap)

    global TAG
    TAG = args.tag
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_style()
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.features)
    if args.sample == "matched":
        df = df[df.matched == 1].copy()
    print(f"author-windows: {len(df):,}")

    # ---------------- 1. feature hygiene ----------------
    print("\nfeature hygiene")
    feat_cols = [c for c in df.columns
                 if c not in NON_FEATURES and block_of(c) in BLOCKS]
    dropped = [c for c in feat_cols if c in DROP_FEATURES]
    for c in dropped:
        print(f"  dropped {c:<22} {DROP_FEATURES[c]}")
    feat_cols = [c for c in feat_cols if c not in DROP_FEATURES]

    if args.drop_counts:
        removed = [c for c in feat_cols if c in COUNT_SCALING]
        print(f"  --drop-counts: removing {len(removed)} exposure-scaling "
              f"features, keeping rates and proportions only")
        for c in removed:
            print(f"    {c}")
        feat_cols = [c for c in feat_cols if c not in COUNT_SCALING]

    if args.drop_partial:
        removed = [c for c in feat_cols if c in PARTIAL_COVERAGE_FEATURES]
        print(f"  --drop-partial: removing {len(removed)} features defined only "
              f"on the thread-completed subset")
        for c in removed:
            print(f"    {c}")
        feat_cols = [c for c in feat_cols if c not in PARTIAL_COVERAGE_FEATURES]

    X_raw = df[feat_cols]
    flagged = check_collinearity(X_raw)
    if flagged:
        print("  WARNING, still near-collinear after drops:")
        for f in flagged:
            print(f"    {f}")
    else:
        print("  no remaining feature is a linear combination of the others")

    # ---------------- 2. residualise on volume ----------------
    covars = pd.DataFrame({
        "log_n": np.log1p(df.n_items.to_numpy(float)),
        "log_t": np.log1p(df.tenure_days.to_numpy(float)),
    })
    X = rank_z(residualise(X_raw, covars))
    cols_by_block = {b: [c for c in feat_cols if block_of(c) == b] for b in BLOCKS}
    for b in BLOCKS:
        print(f"  {BLOCK_NAMES[b]:<9} {len(cols_by_block[b])} features")

    # ---------------- 3. separability ----------------
    print("\nseparability, volume removed")
    corr = X.corr(method="spearman")
    within, between, per_block = [], [], {}
    for b in BLOCKS:
        cb = cols_by_block[b]
        w = [abs(corr.loc[i, j]) for i, j in combinations(cb, 2)]
        o = [abs(corr.loc[i, j]) for i in cb for j in feat_cols if block_of(j) != b]
        per_block[b] = (np.array(w), np.array(o))
        within += w
        between += o
    within = np.array([v for v in within if np.isfinite(v)])
    between = np.array([v for v in between if np.isfinite(v)])
    # NOTE: pairwise correlations share features and are not independent, so
    # this p-value is not valid inference and is not reported in the paper.
    # The block-relabelling permutation test in analyze_validity.py is the
    # test of the partition; the rank-biserial effect size is descriptive.
    u, p_u = stats.mannwhitneyu(within, between, alternative="greater")
    effect = 2 * u / (len(within) * len(between)) - 1
    print(f"  within {np.median(within):.3f}  between {np.median(between):.3f}  "
          f"rank-biserial {effect:.3f}  p {p_u:.2e}")
    for b in BLOCKS:
        w, o = per_block[b]
        print(f"    {BLOCK_NAMES[b]:<9} within {np.median(w):.3f}  "
              f"other {np.median(o):.3f}  gap {np.median(w)-np.median(o):+.3f}")

    # ---------------- 4. PC-truncated CCA ----------------
    print("\ncanonical correlation, PC-truncated to 80% of block variance")
    cca_rows = []
    for a, b in combinations(BLOCKS, 2):
        sub = X[cols_by_block[a] + cols_by_block[b]].dropna()
        Pa, _ = pcs_to(sub[cols_by_block[a]].to_numpy(), 0.8)
        Pb, _ = pcs_to(sub[cols_by_block[b]].to_numpy(), 0.8)
        obs, ua, vb = cca_variates(Pa, Pb)
        null = np.array([cca_first(Pa, Pb[rng.permutation(len(Pb))])
                         for _ in range(args.n_perm)])
        p = float((np.sum(null >= obs) + 1) / (args.n_perm + 1))

        load_a = structure_coefficients(ua, sub[cols_by_block[a]])
        load_b = structure_coefficients(vb, sub[cols_by_block[b]])

        cca_rows.append({"a": a, "b": b, "n": len(sub),
                         "ka": Pa.shape[1], "kb": Pb.shape[1],
                         "rho1": obs, "null_mean": float(null.mean()),
                         "null_p95": float(np.percentile(null, 95)), "p": p,
                         "loadings_a": load_a, "loadings_b": load_b})
        print(f"  {a}-{b}  n={len(sub):,}  k=({Pa.shape[1]},{Pb.shape[1]})  "
              f"rho1 {obs:.3f}  null {null.mean():.3f}  "
              f"p95 {np.percentile(null,95):.3f}  p {p:.4f}")
        print(f"      {a}: " + ", ".join(f"{c.split('_',1)[1]} {r:+.2f}"
                                         for c, r in load_a[:4]))
        print(f"      {b}: " + ", ".join(f"{c.split('_',1)[1]} {r:+.2f}"
                                         for c, r in load_b[:4]))
    for r, pa in zip(cca_rows, holm([r["p"] for r in cca_rows])):
        r["p_holm"] = pa

    # ---------------- 5. estimator validation ----------------
    print("\noffset estimator validation")
    trusted = df[df.P_rhythm_trough_depth <= args.offset_trust]
    od = offset_distribution(trusted)
    print(f"  offsets vs uniform: chi2 {od['chi2']:.1f}, p {od['p']:.3e}")
    print(f"  share in UTC-8..+2 band: {100*od['band_share']:.1f}%")

    rec_rows = []
    if not args.skip_records and os.path.exists(args.db):
        print("\nrecord-level convergence, within author")
        offsets = {r.author_hash: int(r.P_utc_offset_est)
                   for r in trusted.itertuples()
                   if np.isfinite(r.P_utc_offset_est)}
        rec_rows = record_level_convergence(args.db, offsets, rng)

    # ---------------- outputs ----------------
    print("\nfigures")
    fig5b_within_between(within, between, per_block)
    fig6b_canonical(cca_rows)
    fig3_offsets(od)
    if rec_rows:
        fig7b_record_level(rec_rows)

    results = {
        "version": 2, "sample": args.sample, "n_authors": len(df),
        "drop_counts": bool(args.drop_counts),
        "drop_partial": bool(args.drop_partial),
        "dropped_features": DROP_FEATURES,
        "residualised_on": ["log record count", "log tenure"],
        "collinearity_flags": flagged,
        "separability": {"within_med": float(np.median(within)),
                         "between_med": float(np.median(between)),
                         "effect": float(effect), "p": float(p_u),
                         "per_block": {b: {
                             "within": float(np.median(per_block[b][0])),
                             "other": float(np.median(per_block[b][1]))}
                             for b in BLOCKS}},
        "cca": cca_rows, "offset_distribution": od,
        "record_level": rec_rows,
    }
    with open(os.path.join(OUT_DIR, f"results_v2{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # ---------------- verdict ----------------
    print("\n" + "=" * 66)
    print("READING THE RESULT")
    print("=" * 66)
    gap = np.median(within) - np.median(between)
    print(f"Separability gap after removing volume: {gap:+.3f} "
          f"(rank-biserial {effect:+.3f})")
    weak = [BLOCK_NAMES[b] for b in BLOCKS
            if np.median(per_block[b][0]) - np.median(per_block[b][1]) < 0.02]
    if weak:
        print(f"Weakly separable: {', '.join(weak)}")
    above = [f"{r['a']}-{r['b']}" for r in cca_rows if r["rho1"] > r["null_p95"]]
    print(f"Block pairs sharing structure beyond the null: "
          f"{', '.join(above) if above else 'none'}")
    if any(r["rho1"] > 0.95 for r in cca_rows):
        print("WARNING: a canonical correlation is still near 1. Check the")
        print("collinearity flags above; an algebraic dependency remains.")
    print(f"\nOffset estimator: {100*od['band_share']:.0f}% of authors fall in the")
    print("UTC-8 to +2 band, against 44% expected if offsets were uniform.")
    if rec_rows:
        tests = [r for r in rec_rows if r["kind"] == "test"]
        ctrl = [r for r in rec_rows if r["kind"] == "control"]
        sig = [r for r in tests if r["p_holm"] < 0.05 and r["or"] > 1]
        bad = [r for r in ctrl if r["p_holm"] < 0.05 and
               not (0.9 < r["or"] < 1.1)]
        print(f"Record level: {len(sig)}/{len(tests)} probes elevated at night "
              f"after correction (Mantel-Haenszel, author-stratified).")
        for r in sig:
            print(f"  {r['label']}: OR {r['or']:.2f} "
                  f"[{r['or_lo']:.2f}, {r['or_hi']:.2f}]")
        if bad:
            print(f"  controls also moved: {[r['label'] for r in bad]}. "
                  "Investigate before claiming validation.")
        elif sig:
            print("  controls within 0.9-1.1. The offset estimator is validated:")
            print("  the hours it labels local night are when people write about")
            print("  not sleeping, and nothing else moves.")
    print("=" * 66)
    print(f"\noutputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
