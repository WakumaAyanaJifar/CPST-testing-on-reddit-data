#!/usr/bin/env python3
"""
Second-round revision analyses (supervisor review, 2026-10-09).

    T1.1  CPST vs. data channel. Each CPST space maps almost one-to-one onto a
          data channel, so a random-partition null cannot separate CPST from
          grouping by data source. This script
            (a) computes the margin of a pure channel partition next to CPST,
            (b) runs a channel-restricted permutation null (labels permuted
                only among features from the same channel), and
            (c) tests the cross-channel features directly: does each behave
                like its CPST space or like its data channel?
    T2.1  Main separability test with shared-component pairs excluded
          (count/rate pairs, token-length family, nested night windows).
    T2.4  Redundancy in the conservative specification (Table 5 uses it for
          canonical correlations; redundancy was from the full specification).
    T2.5  Best data-driven k = 4 partition margin, as an equal-size benchmark.
    Lex   How many distinct discovery phrases actually matched (39 vs 45).

Read-only. Usage (from the package folder):
    python scripts\\revision2_channel.py
    python scripts\\revision2_channel.py --skip-db      # no lexicon count
Paste the block under "SUMMARY FOR REVISION 2" into the chat.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_validity as AV          # noqa: E402
import analyze_spaces_v2 as AS2        # noqa: E402
import revision_analyses as RA         # noqa: E402
import mh_robustness as MH             # noqa: E402

OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


# ----------------------------------------------------------------------
# data channels (what each feature is computed from)
# ----------------------------------------------------------------------
LINK = {"C_n_link_domains", "C_social_link_share"}            # parsed from record content
REPLY_TIMING = {"P_reply_latency_med", "P_fast_reply_share"}  # timestamps + thread structure


def channel_of(c, reply_to="thread"):
    if c in LINK:
        return "content"
    if c in REPLY_TIMING:
        return reply_to
    return {"C": "metadata", "P": "time", "S": "thread", "T": "content"}[AV.block_of(c)]


# shared-component pairs (Pearson 1897): rate = count / exposure, with the
# exposure residualized out, or nested / derived definitions
LINKED_PAIRS = [
    ("P_active_days", "P_activity_density"), ("P_active_days", "P_items_per_active_day"),
    ("P_activity_density", "P_items_per_active_day"),
    ("P_n_sessions", "P_items_per_session"), ("P_night_share", "P_latenight_share"),
    ("T_tokens_total", "T_mean_tokens"), ("T_tokens_total", "T_median_tokens"),
    ("T_tokens_total", "T_len_variability"), ("T_mean_tokens", "T_median_tokens"),
    ("T_mean_tokens", "T_len_variability"), ("T_median_tokens", "T_len_variability"),
    ("C_n_communities", "C_communities_per_month"),
    ("S_n_partners", "S_partners_per_item"), ("S_replies_received", "S_reply_ratio"),
    ("S_n_threads", "S_items_per_thread"), ("S_mutual_partners", "S_reciprocity"),
    ("S_n_partners", "S_mutual_partners"),
]


def margin_excl(corr, lab, cols, excl):
    w, b = [], []
    for i, j in combinations(cols, 2):
        if frozenset((i, j)) in excl:
            continue
        r = abs(corr.at[i, j])
        if np.isfinite(r):
            (w if lab[i] == lab[j] else b).append(r)
    return float(np.median(w) - np.median(b))


def perm_null(corr, lab, cols, excl, n, rng, within=None):
    """Random relabelling. If `within` maps feature -> stratum, labels are
    permuted only inside each stratum (channel-restricted null)."""
    labs = np.array([lab[c] for c in cols], dtype=object)
    null = np.empty(n)
    for k in range(n):
        if within is None:
            perm = rng.permutation(labs)
        else:
            perm = labs.copy()
            for s in set(within[c] for c in cols):
                idx = [i for i, c in enumerate(cols) if within[c] == s]
                perm[idx] = rng.permutation(labs[idx])
        null[k] = margin_excl(corr, dict(zip(cols, perm)), cols, excl)
    return null


def pval(null, obs):
    return float((np.sum(null >= obs - 1e-12) + 1) / (len(null) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--skip-db", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(os.path.join(AV.BASE_DIR, "cpst_features.csv"))
    cols = [c for c in df.columns if c not in AV.NON_FEATURES
            and AV.block_of(c) in AV.BLOCKS and c not in AV.DROP_FEATURES]
    X, corr = RA.prep(df, cols)
    rel = AV.reliability(pd.read_csv(os.path.join(AV.BASE_DIR, "cpst_features_h1.csv")),
                         pd.read_csv(os.path.join(AV.BASE_DIR, "cpst_features_h2.csv")), cols)
    relmap = dict(zip(rel.feature, rel.r_sb))
    cpst = {c: AV.block_of(c) for c in cols}
    none = set()
    res = {}

    # ---------------- T1.1 ----------------
    say("=== T1.1  CPST versus data channel ===")
    chan_t = {c: channel_of(c, "thread") for c in cols}
    chan_m = {c: channel_of(c, "time") for c in cols}
    differ = [c for c in cols if cpst[c] != {"metadata": "C", "time": "P", "thread": "S",
                                              "content": "T"}[chan_t[c]]]
    say(f"  features where CPST and channel disagree: {len(differ)} of {len(cols)}: {differ}")
    m_cpst = margin_excl(corr, cpst, cols, none)
    m_ct = margin_excl(corr, chan_t, cols, none)
    m_cm = margin_excl(corr, chan_m, cols, none)
    null_rand = perm_null(corr, cpst, cols, none, args.n_perm, rng)
    say(f"  margin  CPST {m_cpst:+.4f}   channel (reply timing=thread) {m_ct:+.4f}   "
        f"channel (reply timing=time) {m_cm:+.4f}")
    say(f"  random null p95 {np.percentile(null_rand, 95):+.4f}  "
        f"(CPST p {pval(null_rand, m_cpst):.4f}; channel p {pval(null_rand, m_ct):.4f})")
    # channel-restricted null: permute CPST labels only within each channel
    null_ch = perm_null(corr, cpst, cols, none, args.n_perm, rng, within=chan_t)
    say(f"  channel-restricted null: mean {null_ch.mean():+.4f}  p95 {np.percentile(null_ch, 95):+.4f}  "
        f"CPST p {pval(null_ch, m_cpst):.4f}")
    say("    (only labels inside a channel move, so this asks whether CPST's placement of the")
    say("     cross-channel features beats other placements within the same channels)")
    # direct test of each cross-channel feature
    say("  cross-channel features: median |r| (and corrected) with each candidate block")
    xt = {}
    for c in differ + ["T_sleep_lex", "T_somatic_lex"]:
        if c not in cols:
            continue
        row = {}
        for blk in AV.BLOCKS:
            others = [o for o in cols if o != c and cpst[o] == blk and o not in differ]
            v = RA.abs_vals(corr, [(c, o) for o in others])
            d = RA.dis_vals(corr, [(c, o) for o in others], relmap)
            row[blk] = (float(np.median(v)), float(np.median(d)) if len(d) else np.nan)
        best = max(row, key=lambda b: row[b][1] if np.isfinite(row[b][1]) else row[b][0])
        xt[c] = {"by_block": row, "closest": best, "cpst": cpst[c]}
        say(f"    {c:<24} CPST {cpst[c]}  " + "  ".join(
            f"{b} {row[b][0]:.3f}/{row[b][1]:.3f}" for b in AV.BLOCKS) + f"   closest {best}")
    res["T1_1"] = {"margin_cpst": m_cpst, "margin_channel_thread": m_ct,
                   "margin_channel_time": m_cm, "random_p95": float(np.percentile(null_rand, 95)),
                   "channel_null_p95": float(np.percentile(null_ch, 95)),
                   "p_cpst_channel_null": pval(null_ch, m_cpst), "cross_channel": xt}

    # ---------------- T2.1 ----------------
    say("\n=== T2.1  separability with shared-component pairs excluded ===")
    excl = {frozenset(p) for p in LINKED_PAIRS if p[0] in cols and p[1] in cols}
    say(f"  excluded pairs: {len(excl)}")
    m_ex = margin_excl(corr, cpst, cols, excl)
    null_ex = perm_null(corr, cpst, cols, excl, args.n_perm, rng)
    say(f"  margin {m_ex:+.4f} (published {m_cpst:+.4f})   random p95 {np.percentile(null_ex, 95):+.4f}   "
        f"p {pval(null_ex, m_ex):.4f}")
    for blk in AV.BLOCKS:
        cb = [c for c in cols if cpst[c] == blk]
        w = [(i, j) for i, j in combinations(cb, 2) if frozenset((i, j)) not in excl]
        o = [(i, j) for i in cb for j in cols if cpst[j] != blk]
        wo, oo = RA.abs_vals(corr, w), RA.abs_vals(corr, o)
        wd, od = RA.dis_vals(corr, w, relmap), RA.dis_vals(corr, o, relmap)
        say(f"    {blk}  gap {np.median(wo) - np.median(oo):+.3f}   gap* {np.median(wd) - np.median(od):+.3f}"
            f"   (within pairs {len(w)})")
    m_ch_ex = margin_excl(corr, chan_t, cols, excl)
    say(f"  channel margin with the same exclusions {m_ch_ex:+.4f}")
    res["T2_1"] = {"margin": m_ex, "p": pval(null_ex, m_ex), "channel_margin": m_ch_ex}

    # ---------------- T2.4 ----------------
    say("\n=== T2.4  redundancy, conservative specification ===")
    cons = [c for c in cols if c not in AS2.COUNT_SCALING and c not in AS2.PARTIAL_COVERAGE_FEATURES]
    by = {b: [c for c in cons if AV.block_of(c) == b] for b in AV.BLOCKS}
    red = {}
    for a, b in combinations(AV.BLOCKS, 2):
        sub = X[by[a] + by[b]].dropna()
        Pa, _ = AS2.pcs_to(sub[by[a]].to_numpy(), 0.8)
        Pb, _ = AS2.pcs_to(sub[by[b]].to_numpy(), 0.8)
        rho, u, v = AS2.cca_variates(Pa, Pb)
        s = sub.reset_index(drop=True)
        vb_a = np.mean([pd.Series(u).corr(s[c]) ** 2 for c in by[b]])   # B explained by A's variate
        va_b = np.mean([pd.Series(v).corr(s[c]) ** 2 for c in by[a]])
        red[f"{a}-{b}"] = {"rho1": rho, f"var({b}|{a})": vb_a, f"var({a}|{b})": va_b}
        say(f"  {a}-{b}  rho1 {rho:.3f}   var({b}|{a}) {vb_a:.3f}   var({a}|{b}) {va_b:.3f}")
    allv = [x for d in red.values() for k, x in d.items() if k.startswith("var")]
    say(f"  range {min(allv) * 100:.1f}% to {max(allv) * 100:.1f}% (full specification: 0.9% to 6.0%)")
    res["T2_4"] = red

    # ---------------- T2.5 ----------------
    say("\n=== T2.5  best data-driven partition with k = 4 ===")
    C = corr.loc[cols, cols].abs().fillna(0).to_numpy().copy()
    np.fill_diagonal(C, 1.0)
    cond = squareform(1 - C, checks=False)
    links = {"average": linkage(cond, "average"), "complete": linkage(cond, "complete"),
             "ward": linkage(X[cols].fillna(0).to_numpy(float).T, "ward")}
    truth = [cpst[c] for c in cols]
    best = None
    for name, Z in links.items():
        lab = fcluster(Z, t=4, criterion="maxclust")
        sizes = np.bincount(lab)[1:]
        m = margin_excl(corr, dict(zip(cols, lab)), cols, none)
        deg = sizes.max() / len(cols) > 0.80
        say(f"  {name:<8} sizes {sorted(sizes.tolist(), reverse=True)}  margin {m:+.4f}  "
            f"ARI {adjusted_rand_score(truth, lab):.3f}{'  (degenerate)' if deg else ''}")
        if not deg and (best is None or m > best[1]):
            best = (name, m)
    if best:
        say(f"  best non-degenerate k=4: {best[0]} margin {best[1]:+.4f}  "
            f"vs CPST {m_cpst:+.4f} and channel {m_ct:+.4f}")
    res["T2_5"] = {"best": best}

    # ---------------- lexicon ----------------
    if not args.skip_db and os.path.exists(MH.DB_PATH):
        say("\n=== Lexicon: discovery phrases that matched ===")
        conn = sqlite3.connect(MH.DB_PATH)
        n_kw = conn.execute("SELECT COUNT(DISTINCT keyword) FROM keyword_hits").fetchone()[0]
        n_dom = conn.execute("SELECT COUNT(DISTINCT domain) FROM keyword_hits").fetchone()[0]
        per = conn.execute("SELECT domain, COUNT(DISTINCT keyword) FROM keyword_hits "
                           "GROUP BY domain").fetchall()
        conn.close()
        say(f"  distinct phrases with at least one hit: {n_kw} in {n_dom} domains: {per}")
        say("  (the collector defines 45 phrases; the paper says 39)")
        res["lexicon"] = {"phrases_hit": n_kw, "domains": n_dom, "per_domain": per}

    with open(os.path.join(AV.OUT_DIR, "revision2_channel.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print("\n" + "=" * 70 + "\nSUMMARY FOR REVISION 2 (paste this into the chat)\n" + "=" * 70)
    print("\n".join(OUT))
    print("=" * 70)


if __name__ == "__main__":
    main()
