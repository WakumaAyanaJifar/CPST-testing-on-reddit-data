#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robustness of the circadian validation
======================================

Two objections to the record-level convergence result, both of which need
answering before submission.

  1. NON-INDEPENDENCE WITHIN AUTHOR.
     The Mantel-Haenszel estimator stratifies by author, which removes every
     between-author confound, but within a stratum it treats records as
     independent Bernoulli draws. They are not: a person writing about not
     sleeping at 03:00 often writes about it again twenty minutes later. Serial
     correlation inflates the effective sample size, so the Robins-Breslow-
     Greenland interval is too narrow. The fix is a cluster bootstrap that
     resamples AUTHORS with replacement, which preserves whatever dependence
     exists inside each author's records.

  2. THE 04:00 ANCHOR IS AN ASSUMPTION.
     The estimator assumes the activity trough is centred at 04:00 local,
     following Della Negra et al. (arXiv 2026), who report that across
     virtually all time zones the activity minimum falls close to that hour.
     The anchor must be an integer: a half-integer value puts the offset
     expression permanently on a tie, and round-half-to-even then makes only
     even offsets reachable. A result
     that survives only at one anchor is an artifact of that anchor, so this
     sweeps the anchor and the night-window definition and reports the odds
     ratio for each.

  3. THE TEST MIGHT ONLY SHOW "SLEEP WORDS IN QUIET HOURS".
     The offset places each author's own activity trough at night, so the
     night window is that author's quietest period by construction. The
     placebo sweep slides a five-hour window around the whole local day and
     contrasts it with the remaining hours. If the estimator is locating sleep,
     the sleep odds ratio should peak at the hours labelled night and fall
     below 1 on the opposite side of the clock; if it merely reflects quiet
     hours, the curve will be flat or peak elsewhere.

  4. CLOCK-TIME TOKENS.
     The sleep lexicon contains "2am", "3am" and "4am", which people write
     when stating the current time. The "sleep_noclock" probe removes them so
     the result cannot be driven by time-stamping in the text itself.

  The analysis is restricted to the author-windows in the feature file, the
  same population as every other result, and the hour histogram uses every
  in-window record, as the feature extractor does. At the 04:00 anchor the
  offsets are therefore identical to P_utc_offset_est, and the script checks
  this; the paper-specification cell must reproduce Table VII exactly.

Outputs
    <BASE_DIR>/paper1/mh_robustness.json
    <BASE_DIR>/paper1/table8_mh_robustness.tex
    <BASE_DIR>/paper1/fig11_anchor_sensitivity.{pdf,png}
    <BASE_DIR>/paper1/fig12_placebo_shift.{pdf,png}

Usage
    python mh_robustness.py
    python mh_robustness.py --n-boot 2000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Data folder: set CPST_BASE_DIR to override the default location.
BASE_DIR = os.environ.get(
    "CPST_BASE_DIR",
    os.path.join(os.path.expanduser("~"), "Documents", "CPST_FOMO_v6"))
DB_PATH = os.path.join(BASE_DIR, "cpst_fomo.db")
FEATURES = os.path.join(BASE_DIR, "cpst_features.csv")
OUT_DIR = os.path.join(BASE_DIR, "paper1")

WINDOW_START_TS = datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp()
WINDOW_END_TS = datetime(2025, 12, 31, tzinfo=timezone.utc).timestamp()

TROUGH_TRUST = 0.75
MIN_PER_CELL = 10

SLEEP_RE = re.compile(
    r"\b(sleep|asleep|insomnia|awake|3am|4am|2am|all night|couldn't sleep|"
    r"stayed up|tired)\b", re.I)
SOMATIC_RE = re.compile(
    r"\b(heart racing|chest|breathe|breathing|nausea|nauseous|"
    r"sick to my stomach|shaking|dizzy|exhausted|headache)\b", re.I)
SLEEP_NOCLOCK_RE = re.compile(
    r"\b(sleep|asleep|insomnia|awake|all night|couldn't sleep|"
    r"stayed up|tired)\b", re.I)
COMPULSION_RE = re.compile(
    r"\b(keep checking|can't stop|refresh|scrolling|again and again|"
    r"every few minutes|compulsive|constantly checking)\b", re.I)
QUESTION_RE = re.compile(r"\?")
SECOND_RE = re.compile(r"\b(you|your|yours|yourself)\b", re.I)

PROBES = [("sleep", SLEEP_RE, "test"),
          ("sleep_noclock", SLEEP_NOCLOCK_RE, "test"),
          ("somatic", SOMATIC_RE, "test"),
          ("compulsion", COMPULSION_RE, "test"),
          ("question", QUESTION_RE, "control"),
          ("second", SECOND_RE, "control")]

# anchor (local hour the trough is assumed to centre on), label
# Integer anchors only. A half-integer anchor makes every odd UTC offset
# unreachable (round-half-to-even), so 03:30 or 04:30 would test a broken
# estimator rather than a different assumption.
ANCHORS = [(2.0, "02:00"), (3.0, "03:00"), (4.0, "04:00 (paper)"),
           (5.0, "05:00"), (6.0, "06:00")]
PLACEBO_WIDTH = 5          # hours in the sliding window
# night window, day window
WINDOWS = [((0, 5), (9, 22), "00-05 vs 09-22"),
           ((1, 4), (9, 22), "01-04 vs 09-22"),
           ((0, 6), (10, 20), "00-06 vs 10-20")]

INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8985"
SURFACE, GRID = "#ffffff", "#e6e5e1"
C_TEST, C_CTRL = "#eb6834", "#8a8985"


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
        "legend.frameon": False, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def estimate_offset(hour_counts: np.ndarray, anchor: float) -> Tuple[int, float]:
    """Same estimator as the extractor, with the anchor exposed."""
    if hour_counts.sum() == 0:
        return 0, 1.0
    wrapped = np.concatenate([hour_counts, hour_counts[:6]])
    win = np.array([wrapped[i:i + 6].sum() for i in range(24)])
    trough_start = int(np.argmin(win))
    trough_centre = (trough_start + 3) % 24
    offset = int(round((anchor - trough_centre) % 24))
    if offset > 12:
        offset -= 24
    mean_6h = hour_counts.sum() / 4.0
    depth = float(win[trough_start] / mean_6h) if mean_6h > 0 else 1.0
    return offset, depth


def mantel_haenszel(t: np.ndarray) -> Dict:
    a, b, c, d = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
    n = a + b + c + d
    keep = n > 0
    a, b, c, d, n = a[keep], b[keep], c[keep], d[keep], n[keep]
    R, S = a * d / n, b * c / n
    sR, sS = R.sum(), S.sum()
    if sR <= 0 or sS <= 0:
        return {"or": np.nan, "lo": np.nan, "hi": np.nan}
    orr = sR / sS
    P, Q = (a + d) / n, (b + c) / n
    var = (np.sum(P * R) / (2 * sR ** 2)
           + np.sum(P * S + Q * R) / (2 * sR * sS)
           + np.sum(Q * S) / (2 * sS ** 2))
    se = np.sqrt(var)
    return {"or": float(orr),
            "lo": float(np.exp(np.log(orr) - 1.96 * se)),
            "hi": float(np.exp(np.log(orr) + 1.96 * se))}


def mh_point(t: np.ndarray) -> float:
    a, b, c, d = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
    n = a + b + c + d
    keep = n > 0
    if not keep.any():
        return np.nan
    a, b, c, d, n = a[keep], b[keep], c[keep], d[keep], n[keep]
    sR, sS = (a * d / n).sum(), (b * c / n).sum()
    return float(sR / sS) if sS > 0 else np.nan


def cluster_bootstrap(t: np.ndarray, n_boot: int, rng) -> Tuple[float, float]:
    """Resample AUTHORS (strata) with replacement. Whatever dependence exists
    inside an author's records is carried along intact, so the interval does
    not assume independence within stratum."""
    m = len(t)
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, m, m)
        out[i] = mh_point(t[idx])
    out = out[np.isfinite(out)]
    if len(out) == 0:                    # probe never occurs in this cell
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def collect(db: str, authors: set | None = None) -> Dict[str, Dict]:
    """One pass: per-author hour histogram and per-(author, hour) probe hits.

    The histogram counts every in-window record, with or without text, which
    is what the feature extractor does; probe hits need text. Restricting to
    the feature file's authors keeps this the same population as Table VII.
    """
    hours = defaultdict(lambda: np.zeros(24, dtype=np.int64))       # all records
    text_hours = defaultdict(lambda: np.zeros(24, dtype=np.int64))  # with text
    hits = defaultdict(lambda: defaultdict(lambda: np.zeros(24, dtype=np.int64)))

    conn = sqlite3.connect(db)
    cur = conn.execute("""
        SELECT author_hash, created_utc, body_raw
        FROM items
        WHERE author_hash IS NOT NULL
          AND created_utc BETWEEN ? AND ?
    """, (WINDOW_START_TS, WINDOW_END_TS))

    seen = 0
    kept = 0
    while True:
        chunk = cur.fetchmany(20000)
        if not chunk:
            break
        for author, ts, text in chunk:
            if authors is not None and author not in authors:
                continue
            h = int(ts // 3600) % 24
            hours[author][h] += 1
            kept += 1
            if not text:
                continue
            text_hours[author][h] += 1
            for name, rx, _kind in PROBES:
                if rx.search(text):
                    hits[author][name][h] += 1
        seen += len(chunk)
        if seen % 200000 == 0:
            print(f"    {seen:,} records")
    conn.close()
    print(f"  kept {kept:,} records for {len(hours):,} authors "
          f"({seen:,} rows read)")
    return {"hours": hours, "text_hours": text_hours, "hits": hits}


def run_cell(data, anchor: float, night: Tuple[int, int], day: Tuple[int, int],
             n_boot: int, rng) -> Dict:
    # Offsets come from ALL records, as in the extractor. Cell sizes come from
    # records WITH TEXT only, because a probe can only hit a record with text;
    # this matches record_level_convergence() in analyze_spaces_v2.py.
    hours, hits, th = data["hours"], data["hits"], data["text_hours"]
    tables = {name: [] for name, _rx, _k in PROBES}
    n_authors = 0

    for author, hc in hours.items():
        off, depth = estimate_offset(hc, anchor)
        if depth > TROUGH_TRUST:
            continue
        # local-hour index: local = (utc + off) % 24  ->  utc = (local - off) % 24
        night_utc = [(h - off) % 24 for h in range(night[0], night[1])]
        day_utc = [(h - off) % 24 for h in range(day[0], day[1])]
        n_night = int(th[author][night_utc].sum())
        n_day = int(th[author][day_utc].sum())
        if n_night < MIN_PER_CELL or n_day < MIN_PER_CELL:
            continue
        n_authors += 1
        for name, _rx, _k in PROBES:
            hh = hits[author][name]
            hn = int(hh[night_utc].sum())
            hd = int(hh[day_utc].sum())
            tables[name].append([hn, n_night - hn, hd, n_day - hd])

    out = {"anchor": anchor, "night": list(night), "day": list(day),
           "n_authors": n_authors, "probes": {}}
    for name, _rx, kind in PROBES:
        t = np.array(tables[name], dtype=float)
        if len(t) == 0:
            continue
        mh = mantel_haenszel(t)
        blo, bhi = cluster_bootstrap(t, n_boot, rng)
        out["probes"][name] = {
            "kind": kind, "or": mh["or"],
            "rbg_lo": mh["lo"], "rbg_hi": mh["hi"],
            "boot_lo": blo, "boot_hi": bhi,
            "ci_widening": float((bhi - blo) / max(mh["hi"] - mh["lo"], 1e-9)),
        }
    return out


def placebo_sweep(data, n_boot: int, rng) -> List[Dict]:
    """Slide a PLACEBO_WIDTH-hour window around the local day (04:00 anchor)
    and contrast it with every other hour, on a fixed set of authors: those
    eligible for the paper-specification cell."""
    hours, hits, th = data["hours"], data["hits"], data["text_hours"]
    offs = {}
    for author, hc in hours.items():
        off, depth = estimate_offset(hc, 4.0)
        if depth > TROUGH_TRUST:
            continue
        night = [(h - off) % 24 for h in range(0, 5)]
        day = [(h - off) % 24 for h in range(9, 22)]
        tc = th[author]
        if tc[night].sum() >= MIN_PER_CELL and tc[day].sum() >= MIN_PER_CELL:
            offs[author] = off
    print(f"  placebo sweep on a fixed set of {len(offs):,} authors")

    rows = []
    for start in range(24):
        local_in = [(start + k) % 24 for k in range(PLACEBO_WIDTH)]
        local_out = [h for h in range(24) if h not in local_in]
        tables = {name: [] for name, _rx, _k in PROBES}
        for author, off in offs.items():
            hc = th[author]
            win = [(h - off) % 24 for h in local_in]
            rest = [(h - off) % 24 for h in local_out]
            n_in, n_out = int(hc[win].sum()), int(hc[rest].sum())
            for name, _rx, _k in PROBES:
                hh = hits[author][name]
                a, c = int(hh[win].sum()), int(hh[rest].sum())
                tables[name].append([a, n_in - a, c, n_out - c])
        row = {"start": start, "probes": {}}
        for name, _rx, kind in PROBES:
            t = np.array(tables[name], dtype=float)
            mh = mantel_haenszel(t)
            lo, hi = cluster_bootstrap(t, n_boot, rng)
            row["probes"][name] = {"kind": kind, "or": mh["or"],
                                   "boot_lo": lo, "boot_hi": hi}
        rows.append(row)
        s_ = row["probes"]["sleep"]
        print(f"    window {start:02d}:00-{(start + PLACEBO_WIDTH) % 24:02d}:00  "
              f"sleep OR {s_['or']:.3f} [{s_['boot_lo']:.3f}, {s_['boot_hi']:.3f}]")
    return rows


def fig12_placebo(rows: List[Dict]):
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    # centre the clock on the paper's night window so the peak is not split
    order = sorted(rows, key=lambda r: ((r["start"] + 12) % 24))
    rows = order
    x = np.array([((r["start"] + 12) % 24) - 12 for r in rows])
    for name, col, lab in (("sleep", C_TEST, "sleep vocabulary"),
                           ("question", C_CTRL, "question marks")):
        o = np.array([r["probes"][name]["or"] for r in rows])
        lo = np.array([r["probes"][name]["boot_lo"] for r in rows])
        hi = np.array([r["probes"][name]["boot_hi"] for r in rows])
        ax.fill_between(x, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(x, o, color=col, lw=1.6, label=lab)
    ax.axhline(1.0, color=INK_2, lw=0.8, ls=":")
    ax.axvline(0, color=INK_2, lw=0.8, ls="--")
    ax.text(0.4, ax.get_ylim()[1], "paper night\nwindow", fontsize=6,
            color=INK_2, va="top")
    ax.set_xlim(-12, 11)
    ax.set_xticks(range(-12, 12, 3))
    ax.set_xticklabels([f"{h % 24:02d}" for h in range(-12, 12, 3)])
    ax.set_xlabel("Start of five-hour window, estimated local hour")
    ax.set_ylabel("MH odds ratio, window vs rest")
    ax.legend(loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"fig12_placebo_shift.{ext}"),
                    dpi=600 if ext == "png" else None)
    plt.close(fig)
    print("wrote fig12_placebo_shift.pdf / .png")


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
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--features", default=FEATURES)
    ap.add_argument("--placebo-boot", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = _args(ap)

    if not os.path.exists(args.db):
        sys.exit(f"Database not found: {args.db}")
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_style()
    rng = np.random.default_rng(args.seed)

    if not os.path.exists(args.features):
        sys.exit(f"Feature file not found: {args.features}")
    feat = pd.read_csv(args.features)
    authors = set(feat.author_hash)
    print(f"restricting to {len(authors):,} author-windows in the feature file")

    print("collecting hour histograms and probe hits")
    data = collect(args.db, authors)

    # consistency: at the 04:00 anchor this estimator must reproduce the
    # extractor's offsets, otherwise Tables VII and VIII describe different runs
    if "P_utc_offset_est" in feat.columns:
        ref = dict(zip(feat.author_hash, feat.P_utc_offset_est))
        diff = sum(1 for a, hc in data["hours"].items()
                   if np.isfinite(ref.get(a, np.nan))
                   and estimate_offset(hc, 4.0)[0] != int(ref[a]))
        print(f"  offset mismatches against P_utc_offset_est: {diff}")
        if diff:
            print("  WARNING: the feature file was not built with the current "
                  "extractor. Rerun extract_cpst_features.py --rematch first.")

    results = []
    print("\nanchor and window sweep")
    for anchor, alabel in ANCHORS:
        for night, day, wlabel in WINDOWS:
            r = run_cell(data, anchor, night, day, args.n_boot, rng)
            r["anchor_label"], r["window_label"] = alabel, wlabel
            results.append(r)
            s = r["probes"].get("sleep", {})
            q = r["probes"].get("question", {})
            print(f"  {alabel:<22} {wlabel:<18} n={r['n_authors']:>5}  "
                  f"sleep OR {s.get('or', float('nan')):.3f} "
                  f"[{s.get('boot_lo', float('nan')):.3f}, "
                  f"{s.get('boot_hi', float('nan')):.3f}]  "
                  f"control {q.get('or', float('nan')):.3f}")

    # headline cell: the paper's own specification
    # the paper's specification is the 04:00 anchor (integer, matches
    # Della Negra et al., and avoids the round-half-to-even degeneracy)
    head = next(r for r in results
                if r["anchor"] == 4.0 and r["night"] == [0, 5])

    print("\n" + "=" * 66)
    print("CLUSTER-ROBUST INTERVALS, paper specification (04:00 anchor)")
    print("=" * 66)
    for name, p in head["probes"].items():
        print(f"  {name:<10} ({p['kind']:<7}) OR {p['or']:.3f}   "
              f"RBG [{p['rbg_lo']:.3f}, {p['rbg_hi']:.3f}]   "
              f"bootstrap [{p['boot_lo']:.3f}, {p['boot_hi']:.3f}]   "
              f"widening x{p['ci_widening']:.2f}")

    sleep_all = [r["probes"]["sleep"] for r in results if "sleep" in r["probes"]]
    survives = sum(1 for p in sleep_all if p["boot_lo"] > 1.0)
    print(f"\nSleep OR interval excludes 1 in {survives}/{len(sleep_all)} "
          f"anchor-window combinations.")
    ctrl = [(r["anchor_label"], r["window_label"], n, p)
            for r in results for n, p in r["probes"].items()
            if p["kind"] == "control"]
    excl = [c for c in ctrl if c[3]["boot_lo"] > 1.0 or c[3]["boot_hi"] < 1.0]
    print(f"Control intervals excluding 1: {len(excl)}/{len(ctrl)} probe-cells.")
    for a_, w_, n_, p_ in excl:
        print(f"    {n_:<9} {a_:<14} {w_:<16} OR {p_['or']:.3f} "
              f"[{p_['boot_lo']:.3f}, {p_['boot_hi']:.3f}]")
    ors_c = [c[3]["or"] for c in ctrl]
    print(f"Control ORs range {min(ors_c):.3f} to {max(ors_c):.3f}.")
    print("=" * 66)

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    labels, ors, los, his, kinds = [], [], [], [], []
    for r in results:
        if r["window_label"] != "00-05 vs 09-22":
            continue
        for name in ("sleep", "question"):
            p = r["probes"].get(name)
            if not p:
                continue
            labels.append(f"{r['anchor_label'].split(' ')[0]}")
            ors.append(p["or"]), los.append(p["boot_lo"]), his.append(p["boot_hi"])
            kinds.append(p["kind"])
    y = np.arange(len(ors))
    for i in range(len(ors)):
        col = C_TEST if kinds[i] == "test" else C_CTRL
        ax.plot([los[i], his[i]], [y[i], y[i]], color=col, lw=2,
                solid_capstyle="butt")
        ax.plot(ors[i], y[i], "o", color=col, ms=4.5,
                markeredgecolor=SURFACE, markeredgewidth=0.8)
    ax.axvline(1.0, color=INK_2, lw=1, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{l}  {'sleep' if k=='test' else 'control'}"
                        for l, k in zip(labels, kinds)])
    ax.invert_yaxis()
    ax.set_xlabel("Mantel-Haenszel OR, cluster-bootstrap 95% CI")
    ax.grid(axis="x", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("Sensitivity to the circadian anchor", loc="left")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"fig11_anchor_sensitivity.{ext}"),
                    dpi=600 if ext == "png" else None)
    plt.close(fig)
    print("\nwrote fig11_anchor_sensitivity.pdf / .png")

    # ---- table ----
    rows = []
    for name, p in head["probes"].items():
        rows.append(f"{name.capitalize()} & {p['kind']} & {p['or']:.3f} & "
                    f"[{p['rbg_lo']:.3f}, {p['rbg_hi']:.3f}] & "
                    f"[{p['boot_lo']:.3f}, {p['boot_hi']:.3f}] \\\\")
    tex = r"""\begin{table}[!t]
\caption{Cluster-robust intervals for the record-level convergence test.
Robins-Breslow-Greenland intervals assume records are independent within
author; the bootstrap resamples authors with replacement and therefore does
not.}
\label{tab:mhrobust}
\centering
\footnotesize
\begin{tabular}{@{}llr@{\hspace{4pt}}l@{\hspace{4pt}}l@{}}
\toprule
Probe & Role & OR & RBG 95\% CI & Bootstrap 95\% CI \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(os.path.join(OUT_DIR, "table8_mh_robustness.tex"), "w",
         encoding="utf-8").write(tex)
    print("wrote table8_mh_robustness.tex")

    with open(os.path.join(OUT_DIR, "mh_robustness.json"), "w",
              encoding="utf-8") as f:
        json.dump({"anchors": ANCHORS, "windows": [list(w[:2]) + [w[2]]
                                                   for w in WINDOWS],
                   "n_boot": args.n_boot, "results": results}, f, indent=2)
    print("\nplacebo sweep (04:00 anchor, sliding five-hour window)")
    placebo = placebo_sweep(data, args.placebo_boot, rng)
    fig12_placebo(placebo)
    peak = max(placebo, key=lambda r: r["probes"]["sleep"]["or"])
    print(f"  sleep OR peaks in the window starting {peak['start']:02d}:00 local")
    with open(os.path.join(OUT_DIR, "mh_placebo.json"), "w",
              encoding="utf-8") as f:
        json.dump(placebo, f, indent=2)
    print(f"\noutputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
