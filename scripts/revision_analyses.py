#!/usr/bin/env python3
"""
Paper 1 revision analyses (reviewer feedback, workstream B).

    B1  physical sub-blocks        are rhythm / tempo / density / reply timing
                                   coherent sub-spaces of the physical space?
    B2  mapping sensitivity        do conclusions survive moving boundary
                                   features to a neighbouring space?
    B3  half-sampling intervals    uncertainty across authors for the margin,
                                   per-space gaps and canonical correlations
    B4  role-matched known groups  contrast A rerun with submission share added
                                   to the matching variables
    B5  circadian subsample        are the 1,290 validation authors different
                                   from the full 2,595?
    B6  automated-account check    behavioural flags for bot-like accounts, and
                                   the main results without flagged accounts

Reuses the definitions in analyze_validity.py, analyze_spaces_v2.py and
mh_robustness.py, so every number is computed exactly as in the paper.
Read-only: it never modifies the database or the feature files.

Usage (from the folder that contains scripts/):
    python scripts/revision_analyses.py
    python scripts/revision_analyses.py --n-boot 200        # quicker
    python scripts/revision_analyses.py --skip-db           # B1-B4 only

Outputs (in <CPST_BASE_DIR>/paper1/):
    revision_analyses.json          every number printed
    revision_flagged_accounts.csv   hashed IDs flagged by B6 (stays local)
Paste the block printed under "SUMMARY FOR REVISION" back into the chat.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_validity as AV          # noqa: E402
import analyze_spaces_v2 as AS2        # noqa: E402
import mh_robustness as MH             # noqa: E402

BLOCKS = AV.BLOCKS
NAMES = AV.BLOCK_NAMES

# Physical sub-groups, exactly as in Table A.1 of the manuscript.
PHYS_GROUPS = {
    "rhythm": ["P_rhythm_strength", "P_hour_entropy", "P_night_share",
               "P_latenight_share", "P_workhours_share", "P_weekend_share",
               "P_rhythm_trough_depth"],
    "tempo": ["P_median_gap_s", "P_burst_share", "P_n_sessions",
              "P_items_per_session"],
    "density": ["P_active_days", "P_activity_density", "P_items_per_active_day"],
    "reply": ["P_reply_latency_med", "P_fast_reply_share"],
}
# B2 boundary reassignments
VARIANTS = {
    "sleep+somatic -> physical": {"T_sleep_lex": "P", "T_somatic_lex": "P"},
    "reply timing -> social": {"P_reply_latency_med": "S", "P_fast_reply_share": "S"},
    "both moves": {"T_sleep_lex": "P", "T_somatic_lex": "P",
                   "P_reply_latency_med": "S", "P_fast_reply_share": "S"},
}
# B6 heuristic thresholds (printed, so they can be reported)
BOT_RULES = {
    "duplicate_share": 0.50,     # >= half of texts (>= 20 chars) are exact repeats
    "rapid_gap_share": 0.30,     # >= 30% of gaps shorter than 10 seconds
    "max_per_day": 200,          # >= 200 records on a single day
    "flat_rhythm_depth": 0.90,   # no daily trough (with >= 300 records)
}
SUMMARY: list[str] = []


def say(line=""):
    print(line)
    SUMMARY.append(line)


# ----------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------
def abs_vals(corr, pairs):
    v = [abs(corr.at[i, j]) for i, j in pairs]
    return np.array([x for x in v if np.isfinite(x)])


def dis_vals(corr, pairs, relmap):
    out = []
    for i, j in pairs:
        r = abs(corr.at[i, j])
        ri, rj = relmap.get(i, np.nan), relmap.get(j, np.nan)
        if np.isfinite(r) and np.isfinite(ri) and np.isfinite(rj) and ri > 0 and rj > 0:
            out.append(min(r / np.sqrt(ri * rj), 1.0))
    return np.array(out)


def gap_table(corr, labels, cols, relmap):
    """Within- vs between-block medians for any labelling of the features."""
    out = {}
    for lab in sorted(set(labels[c] for c in cols), key=str):
        cb = [c for c in cols if labels[c] == lab]
        w = list(combinations(cb, 2))
        o = [(i, j) for i in cb for j in cols if labels[j] != lab]
        wo, oo = abs_vals(corr, w), abs_vals(corr, o)
        wd, od = dis_vals(corr, w, relmap), dis_vals(corr, o, relmap)
        med = lambda a: float(np.median(a)) if len(a) else np.nan
        out[lab] = {"k": len(cb), "pairs": len(w),
                    "within_obs": med(wo), "between_obs": med(oo),
                    "within_dis": med(wd), "between_dis": med(od)}
        out[lab]["gap_obs"] = out[lab]["within_obs"] - out[lab]["between_obs"]
        out[lab]["gap_dis"] = out[lab]["within_dis"] - out[lab]["between_dis"]
    return out


def perm_test(corr, labels, cols, n_perm, rng):
    """Margin of a labelling against random relabellings of equal sizes."""
    obs = AV.separability_margin(corr, labels, cols)
    lab = np.array([labels[c] for c in cols], dtype=object)
    null = np.empty(n_perm)
    for k in range(n_perm):
        null[k] = AV.separability_margin(corr, dict(zip(cols, rng.permutation(lab))), cols)
    return {"margin": obs, "null_p95": float(np.nanpercentile(null, 95)),
            "p": float((np.sum(null >= obs) + 1) / (n_perm + 1))}


def canon_corrs(X, Y):
    Ux, Uy = AS2.whiten(X), AS2.whiten(Y)
    if Ux.shape[1] == 0 or Uy.shape[1] == 0:
        return np.array([0.0, 0.0])
    s = np.linalg.svd(Ux.T @ Uy, compute_uv=False)
    s = np.clip(s, 0, 1)
    return np.pad(s, (0, max(0, 2 - len(s))))[:2]


def prep(df, cols):
    covars = pd.DataFrame({"log_n": np.log1p(df.n_items.to_numpy(float)),
                           "log_t": np.log1p(df.tenure_days.to_numpy(float))})
    X = AV.rank_z(AV.residualise(df[cols].reset_index(drop=True), covars))
    return X, X.corr(method="spearman")


def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sd) if sd > 0 else np.nan


def ci(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (np.nan, np.nan)


# ----------------------------------------------------------------------
# B1 physical sub-blocks
# ----------------------------------------------------------------------
def b1(corr, cols, relmap, n_perm, rng, res):
    say("\n=== B1  physical sub-blocks ===")
    missing = [c for g in PHYS_GROUPS.values() for c in g if c not in cols]
    if missing:
        say(f"  WARNING: not in feature set, skipped: {missing}")
    groups = {g: [c for c in fs if c in cols] for g, fs in PHYS_GROUPS.items()}
    phys = [c for fs in groups.values() for c in fs]
    lab_in = {c: g for g, fs in groups.items() for c in fs}

    # (a) inside the physical space: do sub-groups beat random groupings?
    g_in = gap_table(corr, lab_in, phys, relmap)
    p_in = perm_test(corr, lab_in, phys, n_perm, rng)
    say("  (a) within the physical space only")
    say(f"      {'group':<9}{'k':>3}{'within':>9}{'within*':>9}{'others':>9}{'others*':>9}{'gap*':>8}")
    for g, d in g_in.items():
        say(f"      {g:<9}{d['k']:>3}{d['within_obs']:>9.3f}{d['within_dis']:>9.3f}"
            f"{d['between_obs']:>9.3f}{d['between_dis']:>9.3f}{d['gap_dis']:>+8.3f}")
    say(f"      grouping margin {p_in['margin']:+.4f}  random p95 {p_in['null_p95']:+.4f}  "
        f"p = {p_in['p']:.4f}")
    say("      (reply timing has 2 features = 1 pair: read its row with caution)")

    # (b) refined partition: C, S, T plus the four physical groups
    lab7 = {c: AV.block_of(c) for c in cols}
    lab7.update({c: f"P:{g}" for c, g in lab_in.items()})
    g7 = gap_table(corr, lab7, cols, relmap)
    p4 = perm_test(corr, {c: AV.block_of(c) for c in cols}, cols, n_perm, rng)
    p7 = perm_test(corr, lab7, cols, n_perm, rng)
    say("  (b) refined partition (physical split into its four groups)")
    say(f"      {'block':<11}{'k':>3}{'gap':>8}{'gap*':>8}")
    for g, d in g7.items():
        say(f"      {g:<11}{d['k']:>3}{d['gap_obs']:>+8.3f}{d['gap_dis']:>+8.3f}")
    say(f"      4-block margin {p4['margin']:+.4f} (p {p4['p']:.4f})   "
        f"refined margin {p7['margin']:+.4f} (p {p7['p']:.4f})")
    res["B1"] = {"inside": g_in, "inside_test": p_in, "refined": g7,
                 "margin_4block": p4, "margin_refined": p7}
    return groups


# ----------------------------------------------------------------------
# B2 mapping sensitivity
# ----------------------------------------------------------------------
def b2(corr, cols, relmap, n_perm, rng, res):
    say("\n=== B2  mapping sensitivity ===")
    base = {c: AV.block_of(c) for c in cols}
    out = {}
    for name, moves in [("as published", {})] + list(VARIANTS.items()):
        lab = dict(base)
        lab.update({c: b for c, b in moves.items() if c in lab})
        g = gap_table(corr, lab, cols, relmap)
        p = perm_test(corr, lab, cols, n_perm, rng)
        gaps = "  ".join(f"{b} {g[b]['gap_dis']:+.3f}" for b in BLOCKS)
        say(f"  {name:<27} margin {p['margin']:+.4f} p {p['p']:.4f}   gap*: {gaps}")
        if not moves:
            say("  (self-check: the paper reports margin +0.0314 and gap* C +0.057 "
                "P +0.019 S +0.180 T +0.040)")
        out[name] = {"test": p, "gaps": g}
    res["B2"] = out


# ----------------------------------------------------------------------
# B3 bootstrap across authors
# ----------------------------------------------------------------------
def b3(df, cols, relmap, groups, n_boot, rng, res):
    say(f"\n=== B3  uncertainty across authors ({n_boot} half-samples) ===")
    cons = [c for c in cols if c not in AS2.COUNT_SCALING and c not in AS2.PARTIAL_COVERAGE_FEATURES]
    cons_by = {b: [c for c in cons if AV.block_of(c) == b] for b in BLOCKS}
    lab = {c: AV.block_of(c) for c in cols}
    phys = [c for fs in groups.values() for c in fs]
    lab_in = {c: g for g, fs in groups.items() for c in fs}

    def stats_once(d):
        X, corr = prep(d, cols)
        row = {"margin": AV.separability_margin(corr, lab, cols)}
        g = gap_table(corr, lab, cols, relmap)
        for b in BLOCKS:
            row[f"gap_{b}"] = g[b]["gap_obs"]
            row[f"gapdis_{b}"] = g[b]["gap_dis"]
        row["phys_grouping"] = AV.separability_margin(corr, lab_in, phys)
        for a, b in combinations(BLOCKS, 2):
            sub = X[cons_by[a] + cons_by[b]].dropna()
            Pa, _ = AS2.pcs_to(sub[cons_by[a]].to_numpy(), 0.8)
            Pb, _ = AS2.pcs_to(sub[cons_by[b]].to_numpy(), 0.8)
            r1, r2 = canon_corrs(Pa, Pb)
            row[f"rho1_{a}{b}"], row[f"rho2_{a}{b}"] = r1, r2
        return row

    point = stats_once(df)
    t0, boots = time.time(), []
    n = len(df)
    # Half-sampling without replacement: resampling WITH replacement duplicates
    # rows, which biases canonical correlations upward and the margin downward.
    # With m = n/2 drawn without replacement, the spread of the estimates
    # approximates the sampling SD of the full-sample estimate.
    for k in range(n_boot):
        boots.append(stats_once(df.iloc[rng.choice(n, n // 2, replace=False)]))
        if k == 4:
            per = (time.time() - t0) / 5
            print(f"    about {per * n_boot / 60:.0f} min for all resamples")
    B = pd.DataFrame(boots)
    out = {}
    say("  95% intervals = estimate +/- 1.96 x half-sample SD (reliabilities held fixed)")
    for key in point:
        sd = float(np.nanstd(B[key], ddof=1))
        out[key] = {"point": point[key], "sd": sd,
                    "lo": point[key] - 1.96 * sd, "hi": point[key] + 1.96 * sd}
    say(f"  overall margin            {point['margin']:+.4f}  [{out['margin']['lo']:+.4f}, {out['margin']['hi']:+.4f}]")
    for b in BLOCKS:
        o, d = out[f"gap_{b}"], out[f"gapdis_{b}"]
        say(f"  {NAMES[b]:<10} gap {o['point']:+.3f} [{o['lo']:+.3f}, {o['hi']:+.3f}]   "
            f"gap* {d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")
    o = out["phys_grouping"]
    say(f"  physical grouping margin  {o['point']:+.4f}  [{o['lo']:+.4f}, {o['hi']:+.4f}]")
    say("  canonical correlations, conservative specification")
    for a, b in combinations(BLOCKS, 2):
        r1, r2 = out[f"rho1_{a}{b}"], out[f"rho2_{a}{b}"]
        say(f"    {a}-{b}  rho1 {r1['point']:.3f} [{r1['lo']:.3f}, {r1['hi']:.3f}]   "
            f"rho2 {r2['point']:.3f} [{r2['lo']:.3f}, {r2['hi']:.3f}]")
    res["B3"] = out


# ----------------------------------------------------------------------
# B4 role-matched known groups
# ----------------------------------------------------------------------
def b4(df, X, cols, rng, res):
    say("\n=== B4  role-matched known groups (contrast A) ===")
    d = df.reset_index(drop=True).copy()
    v = pd.DataFrame({"log_n": np.log1p(d.n_items), "log_t": np.log1p(d.tenure_days),
                      "post": d.C_post_share})
    Z = (v - v.mean()) / v.std(ddof=1)
    seed = d.cohort.eq("seed").to_numpy()
    pair = np.full(len(d), -1)
    pid = 0
    for blk in d.seed_block.unique():
        si = np.where(seed & d.seed_block.eq(blk).to_numpy())[0]
        ci_ = list(np.where(~seed & d.seed_block.eq(blk).to_numpy())[0])
        for s in rng.permutation(si):
            if not ci_:
                break
            dist = np.sqrt(((Z.iloc[ci_].to_numpy() - Z.iloc[s].to_numpy()) ** 2).sum(1))
            j = int(np.argmin(dist))
            if dist[j] <= 0.5:
                pair[s] = pair[ci_[j]] = pid
                pid += 1
                ci_.pop(j)
    m = pair >= 0
    say(f"  matched pairs: {pid} ({100 * pid / seed.sum():.0f}% of seeds); "
        f"matching on log records, log tenure and submission share")
    for name in v.columns:
        say(f"    SMD {name:<6} before {smd(v[name][seed], v[name][~seed]):+.3f}   "
            f"after {smd(v[name][m & seed], v[name][m & ~seed]):+.3f}")
    keep = [c for c in cols if AV.block_of(c) in ("C", "P", "S") and c not in AV.SELECTION_CONFOUND]
    y = seed[m].astype(int)
    g = pair[m]
    out = {"pairs": pid}
    for label, cs in [("C", [c for c in keep if c.startswith("C_")]),
                      ("P", [c for c in keep if c.startswith("P_")]),
                      ("S", [c for c in keep if c.startswith("S_")]),
                      ("C+P+S", keep)]:
        auc, sd = AV.cv_auc(X.loc[m, cs].to_numpy(float), y, groups=g)
        out[label] = {"k": len(cs), "auc": auc, "sd": sd}
        say(f"    {label:<6} k={len(cs):<3} AUC {auc:.3f} (SD {sd:.3f})")
    auc, sd = AV.cv_auc(v.loc[m].to_numpy(float), y, groups=g)
    out["baseline"] = {"auc": auc, "sd": sd}
    say(f"    baseline (log records, log tenure, submission share) AUC {auc:.3f} (SD {sd:.3f})")
    res["B4"] = out


# ----------------------------------------------------------------------
# one database pass for B5 and B6
# ----------------------------------------------------------------------
def db_pass(db, authors):
    hours = defaultdict(lambda: np.zeros(24, np.int64))
    text_hours = defaultdict(lambda: np.zeros(24, np.int64))
    times = defaultdict(list)
    texts = defaultdict(list)
    conn = sqlite3.connect(db)
    cur = conn.execute("""
        SELECT author_hash, created_utc, body_raw FROM items
        WHERE author_hash IS NOT NULL AND created_utc BETWEEN ? AND ?
    """, (MH.WINDOW_START_TS, MH.WINDOW_END_TS))
    n = 0
    while True:
        rows = cur.fetchmany(50000)
        if not rows:
            break
        for a, ts, body in rows:
            if a not in authors:
                continue
            h = int(ts // 3600) % 24
            hours[a][h] += 1
            times[a].append(ts)
            if body:
                text_hours[a][h] += 1
                t = " ".join(body.lower().split())
                if len(t) >= 20:
                    texts[a].append(hashlib.md5(t.encode()).hexdigest()[:12])
            n += 1
    conn.close()
    print(f"  read {n:,} records for {len(hours):,} authors")
    return hours, text_hours, times, texts


def b5(df, hours, text_hours, res):
    say("\n=== B5  circadian validation subsample vs full sample ===")
    elig, fail_trust, fail_count = set(), 0, 0
    for a, hc in hours.items():
        off, depth = MH.estimate_offset(hc, 4.0)
        if depth > MH.TROUGH_TRUST:
            fail_trust += 1
            continue
        night = [(h - off) % 24 for h in range(0, 5)]
        day = [(h - off) % 24 for h in range(9, 22)]
        th = text_hours[a]
        if th[night].sum() < MH.MIN_PER_CELL or th[day].sum() < MH.MIN_PER_CELL:
            fail_count += 1
            continue
        elig.add(a)
    say(f"  eligible {len(elig):,} (paper: 1,290); excluded: no trusted rhythm {fail_trust:,}, "
        f"too few night/day records {fail_count:,}")
    e = df.author_hash.isin(elig).to_numpy()
    cmp = [("log records", np.log1p(df.n_items)), ("log tenure", np.log1p(df.tenure_days)),
           ("seed cohort", df.cohort.eq("seed").astype(float)),
           ("anxiety block", df.seed_block.eq("anxiety").astype(float))]
    for c in ["C_post_share", "P_rhythm_strength", "P_night_share", "P_burst_share",
              "S_reply_ratio", "T_sleep_lex", "T_affect_neg"]:
        if c in df.columns:
            cmp.append((c, df[c]))
    out = {"eligible": len(elig), "fail_trust": fail_trust, "fail_count": fail_count, "smd": {}}
    say(f"  {'variable':<18}{'eligible':>10}{'others':>10}{'SMD':>8}")
    for name, s in cmp:
        s = pd.Series(np.asarray(s, float))
        out["smd"][name] = smd(s[e], s[~e])
        say(f"  {name:<18}{np.nanmean(s[e]):>10.3f}{np.nanmean(s[~e]):>10.3f}{out['smd'][name]:>+8.2f}")
    say("  |SMD| < 0.10 negligible, 0.10-0.30 small, > 0.30 worth reporting")
    res["B5"] = out


def b6(df, X, cols, hours, times, texts, rng, res):
    say("\n=== B6  automated-account check (behavioural) ===")
    rows = []
    for a in df.author_hash:
        t = np.sort(np.asarray(times.get(a, []), float))
        if len(t) < 2:
            continue
        gaps = np.diff(t)
        days = np.floor(t / 86400).astype(int)
        hs = texts.get(a, [])
        dup = 1 - len(set(hs)) / len(hs) if hs else 0.0
        _, depth = MH.estimate_offset(hours[a], 4.0)
        rows.append({"author_hash": a, "n": len(t), "dup_share": dup,
                     "rapid_share": float(np.mean(gaps < 10)),
                     "max_per_day": int(np.bincount(days - days.min()).max()),
                     "depth": depth})
    m = pd.DataFrame(rows)
    f = pd.DataFrame({
        "duplicates": m.dup_share >= BOT_RULES["duplicate_share"],
        "rapid-fire": m.rapid_share >= BOT_RULES["rapid_gap_share"],
        "burst day": m.max_per_day >= BOT_RULES["max_per_day"],
        "no daily rest": (m.depth >= BOT_RULES["flat_rhythm_depth"]) & (m.n >= 300),
    })
    m["n_flags"] = f.sum(1)
    say(f"  rules: {BOT_RULES}")
    for c in f.columns:
        say(f"    {c:<14} {int(f[c].sum()):>5} accounts")
    flagged = set(m.loc[m["n_flags"] >= 2, "author_hash"])
    say(f"  flagged by two or more rules: {len(flagged)} of {len(m):,} accounts")
    m.loc[m["n_flags"] >= 1].to_csv(os.path.join(AV.OUT_DIR, "revision_flagged_accounts.csv"), index=False)
    out = {"flag_counts": {c: int(f[c].sum()) for c in f.columns}, "flagged": len(flagged)}
    if flagged:
        keep = ~df.author_hash.isin(flagged).to_numpy()
        d2 = df[keep].reset_index(drop=True)
        X2, corr2 = prep(d2, cols)
        lab = {c: AV.block_of(c) for c in cols}
        out["margin_without"] = AV.separability_margin(corr2, lab, cols)
        g = gap_table(corr2, lab, cols, {})
        out["gaps_without"] = {b: g[b]["gap_obs"] for b in BLOCKS}
        say(f"  without flagged accounts: margin {out['margin_without']:+.4f}; gaps "
            + "  ".join(f"{b} {g[b]['gap_obs']:+.3f}" for b in BLOCKS))
    else:
        say("  nothing flagged; main results unaffected")
    res["B6"] = out


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(AV.BASE_DIR, "cpst_features.csv"))
    ap.add_argument("--h1", default=os.path.join(AV.BASE_DIR, "cpst_features_h1.csv"))
    ap.add_argument("--h2", default=os.path.join(AV.BASE_DIR, "cpst_features_h2.csv"))
    ap.add_argument("--db", default=MH.DB_PATH)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=500, help="number of half-samples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-db", action="store_true", help="run B1-B4 only")
    args = ap.parse_args()
    os.makedirs(AV.OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.features)
    cols = [c for c in df.columns if c not in AV.NON_FEATURES
            and AV.block_of(c) in BLOCKS and c not in AV.DROP_FEATURES]
    print(f"author-windows {len(df):,}, features {len(cols)}")
    X, corr = prep(df, cols)

    print("\nreliability (as in the paper) ...")
    rel = AV.reliability(pd.read_csv(args.h1), pd.read_csv(args.h2), cols)
    relmap = dict(zip(rel.feature, rel.r_sb))

    res = {}
    groups = b1(corr, cols, relmap, args.n_perm, rng, res)
    b2(corr, cols, relmap, args.n_perm, rng, res)
    b4(df, X, cols, rng, res)
    b3(df, cols, relmap, groups, args.n_boot, rng, res)
    if not args.skip_db:
        if not os.path.exists(args.db):
            sys.exit(f"database not found: {args.db}  (or use --skip-db)")
        print("\nreading the database for B5 and B6 ...")
        hours, text_hours, times, texts = db_pass(args.db, set(df.author_hash))
        b5(df, hours, text_hours, res)
        b6(df, X, cols, hours, times, texts, rng, res)

    with open(os.path.join(AV.OUT_DIR, "revision_analyses.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print("\n" + "=" * 70 + "\nSUMMARY FOR REVISION (paste this into the chat)\n" + "=" * 70)
    print("\n".join(SUMMARY))
    print("=" * 70 + f"\nfull results: {os.path.join(AV.OUT_DIR, 'revision_analyses.json')}")


if __name__ == "__main__":
    main()
