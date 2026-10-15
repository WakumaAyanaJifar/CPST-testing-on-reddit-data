#!/usr/bin/env python3
"""
Final-round revision analyses (supervisor review, 2026-10-14).

  T2-5  Paired author bootstrap of sleep odds-ratio differences:
        individual minus modal offsets, individual minus community offsets
        (same resampled authors for every comparison, fixed seed).
  T2-4  Construction null for the social block: reply targets are permuted
        within each thread, keeping every author's record counts, threads and
        reply depths, and the social features are recomputed with the
        extractor's own code. If the social block's coherence survives this
        null, it reflects who interacts with whom, not count/rate construction.
  T2-7  Held-out replication: R2, R3, R4 (paired interval) and R4' (MTMM)
        re-run separately in the anxiety and disengagement community blocks.
  T3    Positive control: the MTMM computed with the two week-split halves as
        the two "methods", showing what a pass looks like on these data.

Read-only. Needs the database, cpst_features.csv, the alternating-record
halves and the week-split halves (cpst_features_wk_h1/h2.csv).
    python scripts\\revision4_final.py
Paste the block under "SUMMARY FOR REVISION 4" into the chat.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_validity as AV          # noqa: E402
import revision_analyses as RA         # noqa: E402
import revision2_channel as R2         # noqa: E402
import revision3_checks as R3          # noqa: E402
import mh_robustness as MH             # noqa: E402
import extract_cpst_features as EX     # noqa: E402

OUT = []
SEED = 20261014
N_BOOT = int(os.environ.get("N_BOOT", 2000))
N_HALF = int(os.environ.get("N_HALF", 200))
N_SOCIAL_NULL = int(os.environ.get("N_SOCIAL_NULL", 3))
SOCIAL = ["S_reply_to_comment_share", "S_mean_depth", "S_n_partners", "S_partners_per_item",
          "S_replies_received", "S_reply_ratio", "S_mutual_partners", "S_reciprocity",
          "S_n_threads", "S_items_per_thread"]


def say(s=""):
    print(s)
    OUT.append(s)


def gap_of(corr, cols, block, excl=frozenset()):
    lab = {c: AV.block_of(c) for c in cols}
    cb = [c for c in cols if lab[c] == block]
    w = [abs(corr.at[i, j]) for i, j in combinations(cb, 2)
         if frozenset((i, j)) not in excl and np.isfinite(corr.at[i, j])]
    o = [abs(corr.at[i, j]) for i in cb for j in cols if lab[j] != block and np.isfinite(corr.at[i, j])]
    return float(np.median(w)), float(np.median(o)), float(np.median(w) - np.median(o))


def mtmm_criteria(df, h1, h2, plat):
    """Composite MTMM (as in revision3_checks) for a set of authors."""
    d = df.copy()
    d["X_platform_vocab"] = [plat.get(a, np.nan) for a in d.author_hash]
    cols = [c for c in d.columns if c not in AV.NON_FEATURES and AV.block_of(c) in AV.BLOCKS
            and c not in AV.DROP_FEATURES]
    cpst = {c: AV.block_of(c) for c in cols}
    blocks = {
        "C-native": [c for c in cols if cpst[c] == "C" and c not in R2.LINK],
        "P-native": [c for c in cols if cpst[c] == "P" and c not in R2.REPLY_TIMING],
        "S-native": [c for c in cols if cpst[c] == "S"],
        "C-text": ["X_platform_vocab"],
        "P-text": ["T_sleep_lex", "T_somatic_lex"],
        "S-text": ["T_self_pl", "T_other_ref", "T_second_person"],
    }
    allc = sorted({c for v in blocks.values() for c in v})
    Xm, _ = RA.prep(d, allc)
    comp = {k: R3.composite(Xm, v) for k, v in blocks.items()}
    r = lambda a, b: abs(np.corrcoef(comp[a], comp[b])[0, 1])
    out = {}
    for t in "CPS":
        val = r(f"{t}-native", f"{t}-text")
        htmm = max(max(r(f"{t}-text", f"{o}-text"), r(f"{t}-native", f"{o}-native"))
                   for o in "CPS" if o != t)
        out[t] = (val, htmm, val > htmm)
    return out


def main():
    rng = np.random.default_rng(SEED)
    base = AV.BASE_DIR
    df = pd.read_csv(os.path.join(base, "cpst_features.csv"))
    cols = [c for c in df.columns if c not in AV.NON_FEATURES
            and AV.block_of(c) in AV.BLOCKS and c not in AV.DROP_FEATURES]
    h1 = pd.read_csv(os.path.join(base, "cpst_features_h1.csv"))
    h2 = pd.read_csv(os.path.join(base, "cpst_features_h2.csv"))
    X, corr = RA.prep(df, cols)
    excl = frozenset(frozenset(p) for p in R2.LINKED_PAIRS)
    authors = set(df.author_hash)
    res = {}

    # ---------------- database pass (platform vocabulary, communities) -------------
    print("reading the database ...")
    plat_c = defaultdict(lambda: np.zeros(2))
    top_sub = defaultdict(lambda: defaultdict(int))
    sub_hours = defaultdict(lambda: np.zeros(24, np.int64))
    conn = sqlite3.connect(MH.DB_PATH)
    cur = conn.execute("""SELECT author_hash, created_utc, body_raw, subreddit FROM items
                          WHERE author_hash IS NOT NULL AND created_utc BETWEEN ? AND ?""",
                       (MH.WINDOW_START_TS, MH.WINDOW_END_TS))
    while True:
        rows = cur.fetchmany(50000)
        if not rows:
            break
        for a, ts, body, sub in rows:
            sub_hours[sub][int(ts // 3600) % 24] += 1
            if a in authors:
                top_sub[a][sub] += 1
                if body:
                    plat_c[a] += (len(R3.PLATFORM_RE.findall(body)), len(body.split()))
    plat = {a: 100 * v[0] / max(v[1], 1) for a, v in plat_c.items()}

    # ================= T2-5 paired offset bootstrap =================
    say("=== T2-5  paired author bootstrap of sleep odds-ratio differences ===")
    data = MH.collect(MH.DB_PATH, authors)
    th, hits = data["text_hours"], data["hits"]
    ind = {}
    for a, hc in data["hours"].items():
        off, depth = MH.estimate_offset(hc, 4.0)
        if depth > MH.TROUGH_TRUST:
            continue
        ni = [(h - off) % 24 for h in range(0, 5)]
        di = [(h - off) % 24 for h in range(9, 22)]
        if th[a][ni].sum() >= MH.MIN_PER_CELL and th[a][di].sum() >= MH.MIN_PER_CELL:
            ind[a] = off
    elig = sorted(ind)
    vals, cnt = np.unique(list(ind.values()), return_counts=True)
    mode = int(vals[np.argmax(cnt)])
    comm_off = {s: MH.estimate_offset(hc, 4.0)[0] for s, hc in sub_hours.items()}
    offs = {"individual": ind, "modal": {a: mode for a in elig},
            "community": {a: comm_off[max(top_sub[a], key=top_sub[a].get)] for a in elig}}

    def table(o, probe="sleep"):
        t = []
        for a in elig:
            ni = [(h - o[a]) % 24 for h in range(0, 5)]
            di = [(h - o[a]) % 24 for h in range(9, 22)]
            nn, nd = int(th[a][ni].sum()), int(th[a][di].sum())
            hn, hd = int(hits[a][probe][ni].sum()), int(hits[a][probe][di].sum())
            t.append([hn, nn - hn, hd, nd - hd])
        return np.array(t, float)
    T = {k: table(v) for k, v in offs.items()}
    point = {k: MH.mantel_haenszel(t)["or"] for k, t in T.items()}
    n = len(elig)
    boot = {k: [] for k in T}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        for k, t in T.items():
            boot[k].append(MH.mantel_haenszel(t[idx])["or"])
    B = {k: np.array(v) for k, v in boot.items()}
    out = {}
    for k in T:
        lo, hi = np.percentile(B[k], [2.5, 97.5])
        out[k] = {"or": point[k], "lo": lo, "hi": hi}
        say(f"  {k:<11} OR {point[k]:.3f} [{lo:.3f}, {hi:.3f}]")
    for k in ("modal", "community"):
        dd = B["individual"] - B[k]
        lo, hi = np.percentile(dd, [2.5, 97.5])
        p = float(min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean())))
        out[f"individual-{k}"] = {"diff": point["individual"] - point[k], "lo": lo, "hi": hi, "p": p}
        say(f"  individual minus {k:<9} {point['individual'] - point[k]:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
            f"two-sided bootstrap p {p:.4f}")
    say(f"  ({N_BOOT} author resamples, seed {SEED}, {n} authors, modal offset UTC{mode:+d})")
    res["T2_5"] = out

    # ================= T2-4 social construction null =================
    say("\n=== T2-4  social construction null (reply targets permuted within threads) ===")
    w_obs, b_obs, g_obs = gap_of(corr, cols, "S", excl)
    say(f"  observed social block: within {w_obs:.3f}  between {b_obs:.3f}  gap {g_obs:+.3f} (linked pairs excluded)")
    conn = sqlite3.connect(EX.DB_PATH if hasattr(EX, "DB_PATH") else MH.DB_PATH)
    meta = EX.load_metadata(conn, 30)
    conn.close()
    meta = meta[meta.author_hash.isin(authors)].copy()
    nulls = []
    for s in range(N_SOCIAL_NULL):
        r_ = np.random.default_rng(SEED + s)
        m = meta.copy()
        rep = m.parent_id.notna() & m.link_id.notna()
        for _, idx in m[rep].groupby("link_id").groups.items():
            idx = list(idx)
            if len(idx) > 1:
                m.loc[idx, "parent_id"] = r_.permutation(m.loc[idx, "parent_id"].to_numpy())
        feats = EX.cyber_physical_social(m).set_index("author_hash")
        d = df.set_index("author_hash").copy()
        for c in SOCIAL:
            if c in feats.columns and c in d.columns:
                d[c] = feats[c].reindex(d.index)
        Xn, cn = RA.prep(d.reset_index(), cols)
        nulls.append(gap_of(cn, cols, "S", excl))
        print(f"    null run {s + 1}: within {nulls[-1][0]:.3f}  gap {nulls[-1][2]:+.3f}")
    nw = np.mean([x[0] for x in nulls]); ng = np.mean([x[2] for x in nulls])
    say(f"  null ({N_SOCIAL_NULL} runs): within {nw:.3f}  gap {ng:+.3f} "
        f"(range {min(x[2] for x in nulls):+.3f} to {max(x[2] for x in nulls):+.3f})")
    res["T2_4"] = {"observed": [w_obs, b_obs, g_obs], "null": nulls}

    # ================= T2-7 held-out replication by community block =================
    say("\n=== T2-7  held-out replication by community block ===")
    rel = AV.reliability(h1, h2, cols)
    relmap = dict(zip(rel.feature, rel.r_sb))
    cmap = {"metadata": "C", "time": "P", "thread": "S", "content": "T"}
    chan = {c: cmap[R2.channel_of(c)] for c in cols}
    cpst = {c: AV.block_of(c) for c in cols}
    rep_out = {}
    for blk in sorted(df.seed_block.dropna().unique()):
        sub = df[df.seed_block == blk].reset_index(drop=True)
        _, cs = RA.prep(sub, cols)
        m_c = AV.separability_margin(cs, cpst, cols)
        labs = np.array([cpst[c] for c in cols], dtype=object)
        null = [AV.separability_margin(cs, dict(zip(cols, rng.permutation(labs))), cols) for _ in range(500)]
        g = RA.gap_table(cs, cpst, cols, relmap)
        diffs = []
        for _ in range(N_HALF):
            idx = rng.choice(len(sub), len(sub) // 2, replace=False)
            _, cb = RA.prep(sub.iloc[idx].reset_index(drop=True), cols)
            diffs.append(AV.separability_margin(cb, cpst, cols) - AV.separability_margin(cb, chan, cols))
        dmain = m_c - AV.separability_margin(cs, chan, cols)
        sd = float(np.std(diffs, ddof=1))
        mt = mtmm_criteria(sub, h1, h2, plat)
        say(f"  {blk} (n={len(sub)}): R2 margin {m_c:+.4f} vs null p95 {np.percentile(null, 95):+.4f}   "
            f"R3 gap* " + " ".join(f"{b} {g[b]['gap_dis']:+.3f}" for b in AV.BLOCKS))
        say(f"      R4 CPST-channel {dmain:+.4f} [{dmain - 1.96 * sd:+.4f}, {dmain + 1.96 * sd:+.4f}]   "
            f"R4' " + "  ".join(f"{t} {v[0]:.3f} vs {v[1]:.3f} {'pass' if v[2] else 'fail'}" for t, v in mt.items()))
        rep_out[blk] = {"margin": m_c, "p95": float(np.percentile(null, 95)),
                        "gaps": {b: g[b]["gap_dis"] for b in AV.BLOCKS},
                        "r4_diff": [dmain, dmain - 1.96 * sd, dmain + 1.96 * sd],
                        "r4p": {t: list(map(float, v[:2])) + [bool(v[2])] for t, v in mt.items()}}
    res["T2_7"] = rep_out

    # ================= T3 positive control =================
    say("\n=== T3  positive control: MTMM with week halves as the two 'methods' ===")
    w1 = pd.read_csv(os.path.join(base, "cpst_features_wk_h1.csv"))
    w2 = pd.read_csv(os.path.join(base, "cpst_features_wk_h2.csv"))
    common = sorted(set(w1.author_hash) & set(w2.author_hash))
    w1 = w1.set_index("author_hash").loc[common].reset_index()
    w2 = w2.set_index("author_hash").loc[common].reset_index()
    comp = {}
    for tag, w in (("h1", w1), ("h2", w2)):
        Xw, _ = RA.prep(w, cols)
        for b in AV.BLOCKS:
            comp[f"{b}-{tag}"] = R3.composite(Xw, [c for c in cols if cpst[c] == b])
    r = lambda a, b: abs(np.corrcoef(comp[a], comp[b])[0, 1])
    pc = {}
    for b in AV.BLOCKS:
        val = r(f"{b}-h1", f"{b}-h2")
        htmm = max(max(r(f"{b}-h1", f"{o}-h1"), r(f"{b}-h2", f"{o}-h2")) for o in AV.BLOCKS if o != b)
        pc[b] = (val, htmm)
        say(f"  {AV.BLOCK_NAMES[b]:<9} same space across halves {val:.3f}   "
            f"largest different-space within a half {htmm:.3f}   {'pass' if val > htmm else 'fail'}")
    say(f"  ({len(common)} authors present in both week halves)")
    res["T3_positive_control"] = pc

    os.makedirs(AV.OUT_DIR, exist_ok=True)
    with open(os.path.join(AV.OUT_DIR, "revision4_final.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))
    print("\n" + "=" * 70 + "\nSUMMARY FOR REVISION 4 (paste this into the chat)\n" + "=" * 70)
    print("\n".join(OUT))
    print("=" * 70)


if __name__ == "__main__":
    main()
