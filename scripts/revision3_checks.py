#!/usr/bin/env python3
"""
Third-round revision analyses (supervisor review, 2026-10-13).

  T1-1a  Paired half-sampling interval for the CPST-minus-channel margin.
  T1-1b  Multitrait-multimethod matrix with TEXT as a second method for three
         spaces (Campbell & Fiske 1959):
             physical : native = timestamps,       text = sleep, somatic words
             social   : native = thread structure, text = we / other / you words
             cyber    : native = metadata,         text = platform vocabulary
  T1-2   Rhythm/tempo checks: direct rhythm x tempo correlation; rhythm
         coherence without trough depth; offset-invariant vs offset-dependent
         rhythm features; cross-half offset (offset from one half, features from
         the other); and a null in which each record's hour is redrawn from the
         pooled hour distribution (destroys individual rhythm, keeps gaps).
         Note: permuting hours WITHIN an author leaves that author's hour
         histogram unchanged, so it cannot serve as this null.
  T2-1   Block-split reliability, if cpst_features_wk_h1/h2.csv exist
         (made with extract_cpst_features.py --half 1/2 --split week).
  T2-2   Circadian baselines: sleep odds ratio under individual, modal,
         shuffled and community-level offsets, on the same 1290 authors.

Read-only. Usage (from the package folder):
    python scripts\\revision3_checks.py
Paste the block under "SUMMARY FOR REVISION 3" into the chat.
"""
from __future__ import annotations

import json
import math
import os
import re
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
import mh_robustness as MH             # noqa: E402

OUT = []
N_HALF = int(os.environ.get("N_HALF", 500))
RNG = np.random.default_rng(42)


def say(s=""):
    print(s)
    OUT.append(s)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def med_abs(corr, A, B):
    v = [abs(corr.at[i, j]) for i in A for j in B if i != j and np.isfinite(corr.at[i, j])]
    return float(np.median(v)) if v else np.nan


def med_within(corr, A, excl=()):
    v = [abs(corr.at[i, j]) for i, j in combinations(A, 2)
         if frozenset((i, j)) not in excl and np.isfinite(corr.at[i, j])]
    return float(np.median(v)) if v else np.nan


def composite(X, cols):
    """First principal component of a block (residualized rank-z features)."""
    Z = X[cols].fillna(0.0).to_numpy(float)
    if Z.shape[1] == 1:
        return Z[:, 0]
    u, s, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    sc = u[:, 0] * s[0]
    if np.corrcoef(sc, Z.mean(1))[0, 1] < 0:   # orient towards the block mean
        sc = -sc
    return sc


def norm_entropy(counts):
    c = np.asarray(counts, float)
    k = int((c > 0).sum())
    if k <= 1:
        return 0.0
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum() / math.log(k, 2))


def rhythm_from_hours(hours_utc, offset=None):
    """Rhythm features exactly as in extract_cpst_features.py."""
    hc = np.bincount(hours_utc, minlength=24)
    off, depth = MH.estimate_offset(hc, 4.0)
    if offset is not None:
        off = offset
    hl = (hours_utc + off) % 24
    th = 2 * np.pi * hl / 24.0
    return {"P_rhythm_trough_depth": depth,
            "P_rhythm_strength": float(np.hypot(np.cos(th).mean(), np.sin(th).mean())),
            "P_hour_entropy": norm_entropy(np.bincount(hl, minlength=24)),
            "P_night_share": float(((hl >= 0) & (hl < 5)).mean()),
            "P_latenight_share": float(((hl >= 1) & (hl < 4)).mean()),
            "P_workhours_share": float(((hl >= 9) & (hl < 18)).mean())}, off


PLATFORM_RE = re.compile(
    r"\b(reddit|subreddits?|instagram|insta|tiktok|youtube|twitter|facebook|snapchat|"
    r"discord|whatsapp|apps?|feeds?|scroll(?:ing|ed)?|notifications?|phones?|"
    r"smartphones?|screen ?time|online|internet|social media|website|posts?|upvotes?)\b", re.I)
RHYTHM = ["P_rhythm_trough_depth", "P_rhythm_strength", "P_hour_entropy",
          "P_night_share", "P_latenight_share", "P_workhours_share", "P_weekend_share"]
TEMPO = ["P_median_gap_s", "P_burst_share", "P_n_sessions", "P_items_per_session"]


def main():
    base = AV.BASE_DIR
    df = pd.read_csv(os.path.join(base, "cpst_features.csv"))
    cols = [c for c in df.columns if c not in AV.NON_FEATURES
            and AV.block_of(c) in AV.BLOCKS and c not in AV.DROP_FEATURES]
    h1 = pd.read_csv(os.path.join(base, "cpst_features_h1.csv"))
    h2 = pd.read_csv(os.path.join(base, "cpst_features_h2.csv"))
    X, corr = RA.prep(df, cols)
    cpst = {c: AV.block_of(c) for c in cols}
    cmap = {"metadata": "C", "time": "P", "thread": "S", "content": "T"}
    chan = {c: cmap[R2.channel_of(c)] for c in cols}
    res = {}

    # ================= T1-1a paired interval =================
    say("=== T1-1a  CPST minus channel margin, paired half-sampling ===")
    m_c = AV.separability_margin(corr, cpst, cols)
    m_h = AV.separability_margin(corr, chan, cols)
    n = len(df)
    diffs = []
    for _ in range(N_HALF):
        idx = RNG.choice(n, n // 2, replace=False)
        _, cb = RA.prep(df.iloc[idx].reset_index(drop=True), cols)
        diffs.append(AV.separability_margin(cb, cpst, cols) - AV.separability_margin(cb, chan, cols))
    sd = float(np.std(diffs, ddof=1))
    d = m_c - m_h
    say(f"  CPST {m_c:+.4f}  channel {m_h:+.4f}  difference {d:+.4f}  "
        f"95% interval [{d - 1.96 * sd:+.4f}, {d + 1.96 * sd:+.4f}]  ({N_HALF} paired half-samples)")
    res["T1_1a"] = {"cpst": m_c, "channel": m_h, "diff": d, "lo": d - 1.96 * sd, "hi": d + 1.96 * sd}

    # ================= database pass (platform vocabulary, hours, gaps) =================
    print("\nreading the database ...")
    authors = set(df.author_hash)
    hours = defaultdict(list)
    times = defaultdict(list)
    plat = defaultdict(lambda: np.zeros(2))           # matches, tokens (all records)
    plat_h = defaultdict(lambda: np.zeros((2, 2)))    # per alternating half
    rank = defaultdict(int)
    top_sub = defaultdict(lambda: defaultdict(int))
    sub_hours = defaultdict(lambda: np.zeros(24, np.int64))
    conn = sqlite3.connect(MH.DB_PATH)
    cur = conn.execute("""SELECT author_hash, created_utc, body_raw, subreddit FROM items
                          WHERE author_hash IS NOT NULL AND created_utc BETWEEN ? AND ?
                          ORDER BY author_hash, created_utc""", (MH.WINDOW_START_TS, MH.WINDOW_END_TS))
    while True:
        rows = cur.fetchmany(50000)
        if not rows:
            break
        for a, ts, body, sub in rows:
            h = int(ts // 3600) % 24
            sub_hours[sub][h] += 1
            if a not in authors:
                continue
            hours[a].append(h)
            times[a].append(ts)
            top_sub[a][sub] += 1
            if body:
                k = len(PLATFORM_RE.findall(body))
                t = len(body.split())
                plat[a] += (k, t)
                plat_h[a][rank[a] % 2] += (k, t)
            rank[a] += 1
    conn.close()

    # ================= T1-1b MTMM =================
    say("\n=== T1-1b  multitrait-multimethod matrix (text as second method) ===")
    df["X_platform_vocab"] = [100 * plat[a][0] / max(plat[a][1], 1) for a in df.author_hash]
    for hh, hdf in ((0, h1), (1, h2)):
        hdf["X_platform_vocab"] = [100 * plat_h[a][hh][0] / max(plat_h[a][hh][1], 1)
                                   if a in plat_h else np.nan for a in hdf.author_hash]
    blocks = {
        "C-native": [c for c in cols if cpst[c] == "C" and c not in R2.LINK],
        "P-native": [c for c in cols if cpst[c] == "P" and c not in R2.REPLY_TIMING],
        "S-native": [c for c in cols if cpst[c] == "S"],
        "C-text": ["X_platform_vocab"],
        "P-text": ["T_sleep_lex", "T_somatic_lex"],
        "S-text": ["T_self_pl", "T_other_ref", "T_second_person"],
    }
    allc = sorted({c for v in blocks.values() for c in v})
    Xm, cm = RA.prep(df, allc)
    comp = {k: composite(Xm, v) for k, v in blocks.items()}

    def comp_rel(k):
        out = []
        for hdf in (h1, h2):
            Xh, _ = RA.prep(hdf.set_index("author_hash").loc[df.author_hash].reset_index(), blocks[k])
            out.append(composite(Xh, blocks[k]))
        r = np.corrcoef(out[0], out[1])[0, 1]
        return 2 * r / (1 + r)
    rel = {k: comp_rel(k) for k in blocks}
    keys = list(blocks)
    R = pd.DataFrame([[abs(np.corrcoef(comp[a], comp[b])[0, 1]) for b in keys] for a in keys],
                     index=keys, columns=keys)
    say("  |r| between block composites (first principal component); diagonal = split-half reliability")
    say("            " + "".join(f"{k:>10}" for k in keys))
    for a in keys:
        say(f"  {a:<10}" + "".join(f"{(rel[a] if a == b else R.at[a, b]):>10.3f}" for b in keys))
    say("  Campbell-Fiske comparison (composites; * = corrected for unreliability)")
    out = {}
    for t in "CPS":
        val = R.at[f"{t}-native", f"{t}-text"]
        valc = val / math.sqrt(max(rel[f"{t}-native"] * rel[f"{t}-text"], 1e-6))
        htmm_txt = max(R.at[f"{t}-text", f"{o}-text"] for o in "CPS" if o != t)
        htmm_nat = max(R.at[f"{t}-native", f"{o}-native"] for o in "CPS" if o != t)
        hthm = max(max(R.at[f"{t}-native", f"{o}-text"], R.at[f"{t}-text", f"{o}-native"])
                   for o in "CPS" if o != t)
        out[t] = {"validity": val, "validity_corr": valc, "max_HTMM_text": htmm_txt,
                  "max_HTMM_native": htmm_nat, "max_HTHM": hthm}
        say(f"  {t}: same trait, different method {val:.3f} ({valc:.3f}*)   "
            f"vs different trait, same method: text {htmm_txt:.3f}, native {htmm_nat:.3f}   "
            f"vs different trait and method {hthm:.3f}")
    say("  feature-level medians |r| (no composites):")
    for t in "CPS":
        say(f"    {t}: native x text {med_abs(cm, blocks[f'{t}-native'], blocks[f'{t}-text']):.3f}   "
            f"text x other-trait text {np.median([med_abs(cm, blocks[f'{t}-text'], blocks[f'{o}-text']) for o in 'CPS' if o != t]):.3f}")
    res["T1_1b"] = {"composite_abs_r": R.round(4).to_dict(), "reliability": rel, "criteria": out}

    # ================= T1-2 rhythm / tempo =================
    say("\n=== T1-2  rhythm and tempo ===")
    rh = [c for c in RHYTHM if c in cols]
    te = [c for c in TEMPO if c in cols]
    excl = {frozenset(p) for p in R2.LINKED_PAIRS}
    say(f"  rhythm within {med_within(corr, rh, excl):.3f}   tempo within {med_within(corr, te, excl):.3f}   "
        f"rhythm x tempo (direct) {med_abs(corr, rh, te):.3f}")
    rh_nod = [c for c in rh if c != "P_rhythm_trough_depth"]
    inv = ["P_rhythm_trough_depth", "P_rhythm_strength", "P_hour_entropy"]
    dep = ["P_night_share", "P_latenight_share", "P_workhours_share"]
    say(f"  rhythm without trough depth {med_within(corr, rh_nod, excl):.3f}   "
        f"offset-invariant (depth, strength, entropy) {med_within(corr, inv):.3f}   "
        f"offset-dependent (night, late-night, work hours) {med_within(corr, dep, excl):.3f}   "
        f"invariant x dependent {med_abs(corr, inv, dep):.3f}")

    def rhythm_table(hours_map, offsets=None, rng=None, pooled=None):
        rows = []
        for a in df.author_hash:
            hu = np.asarray(hours_map[a], int)
            if pooled is not None:
                hu = rng.choice(24, size=len(hu), p=pooled)
            f, _ = rhythm_from_hours(hu, None if offsets is None else offsets.get(a))
            rows.append(f)
        return pd.DataFrame(rows)

    def coh(tab):
        cov = pd.DataFrame({"log_n": np.log1p(df.n_items.to_numpy(float)),
                            "log_t": np.log1p(df.tenure_days.to_numpy(float))})
        Z = AV.rank_z(AV.residualise(tab.reset_index(drop=True), cov))
        c = Z.corr(method="spearman")
        cc = list(tab.columns)
        return med_within(c, cc, excl)
    base_tab = rhythm_table(hours)
    obs = coh(base_tab)
    # cross-half offset: offset from even-indexed records, features from odd-indexed records
    hA = {a: np.asarray(v[0::2], int) for a, v in hours.items()}
    hB = {a: np.asarray(v[1::2], int) for a, v in hours.items()}
    offA = {a: MH.estimate_offset(np.bincount(hA[a], minlength=24), 4.0)[0] for a in df.author_hash}
    cross = coh(rhythm_table(hB, offsets=offA))
    same = coh(rhythm_table(hB))
    # null: hours redrawn from the pooled distribution (gaps and counts kept)
    pooled = np.bincount(np.concatenate([np.asarray(v, int) for v in hours.values()]), minlength=24)
    pooled = pooled / pooled.sum()
    nulls = [coh(rhythm_table(hours, rng=np.random.default_rng(s), pooled=pooled)) for s in range(5)]
    say(f"  rhythm coherence (6 hour-based features): observed {obs:.3f}   "
        f"half B with own offset {same:.3f}   half B with offset from half A {cross:.3f}")
    say(f"  null with hours redrawn from the pooled distribution: {np.mean(nulls):.3f} "
        f"(range {min(nulls):.3f} to {max(nulls):.3f}, 5 draws)")
    res["T1_2"] = {"rhythm": med_within(corr, rh, excl), "tempo": med_within(corr, te, excl),
                   "rhythm_x_tempo": med_abs(corr, rh, te), "no_depth": med_within(corr, rh_nod, excl),
                   "observed6": obs, "same_half": same, "cross_half": cross, "null": nulls}

    # ================= T2-1 block-split reliability =================
    say("\n=== T2-1  block-split (alternating weeks) reliability ===")
    wk1, wk2 = os.path.join(base, "cpst_features_wk_h1.csv"), os.path.join(base, "cpst_features_wk_h2.csv")
    if os.path.exists(wk1) and os.path.exists(wk2):
        rw = AV.reliability(pd.read_csv(wk1), pd.read_csv(wk2), cols)
        ra = AV.reliability(h1, h2, cols)
        for b in AV.BLOCKS:
            fa = ra[ra.feature.map(AV.block_of) == b].r_sb
            fw = rw[rw.feature.map(AV.block_of) == b].r_sb
            say(f"  {AV.BLOCK_NAMES[b]:<9} median: alternating records {fa.median():.3f}   "
                f"alternating weeks {fw.median():.3f}   (>=0.70: {int((fw >= .7).sum())}/{len(fw)})")
        res["T2_1"] = {b: float(rw[rw.feature.map(AV.block_of) == b].r_sb.median()) for b in AV.BLOCKS}
    else:
        say("  week-split files not found; run first:")
        say("    python scripts\\extract_cpst_features.py --half 1 --split week")
        say("    python scripts\\extract_cpst_features.py --half 2 --split week")

    # ================= T2-2 circadian baselines =================
    say("\n=== T2-2  circadian baselines (sleep odds ratio, same authors) ===")
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
    elig = list(ind)

    def or_for(offs, probe="sleep", boot=1000):
        t = []
        for a in elig:
            off = offs[a]
            ni = [(h - off) % 24 for h in range(0, 5)]
            di = [(h - off) % 24 for h in range(9, 22)]
            nn, nd = int(th[a][ni].sum()), int(th[a][di].sum())
            hh = hits[a][probe]
            hn, hd = int(hh[ni].sum()), int(hh[di].sum())
            t.append([hn, nn - hn, hd, nd - hd])
        t = np.array(t, float)
        mh = MH.mantel_haenszel(t)
        lo, hi = MH.cluster_bootstrap(t, boot, RNG) if boot else (np.nan, np.nan)
        return mh["or"], lo, hi
    vals, cnt = np.unique(list(ind.values()), return_counts=True)
    mode = int(vals[np.argmax(cnt)])
    comm_off = {s: MH.estimate_offset(hc, 4.0)[0] for s, hc in sub_hours.items()}
    community = {a: comm_off[max(top_sub[a], key=top_sub[a].get)] for a in elig}
    rows = [("individual offsets", ind), (f"one modal offset (UTC{mode:+d}) for all", {a: mode for a in elig}),
            ("community-level offsets", community)]
    out = {}
    say(f"  authors: {len(elig)}")
    for name, offs in rows:
        o, lo, hi = or_for(offs)
        q = or_for(offs, "question", 0)[0]
        out[name] = (o, lo, hi, q)
        say(f"  {name:<34} sleep OR {o:.3f} [{lo:.3f}, {hi:.3f}]   question-mark control {q:.3f}")
    sh = []
    offv = np.array([ind[a] for a in elig])
    for s in range(20):
        perm = np.random.default_rng(100 + s).permutation(offv)
        sh.append(or_for(dict(zip(elig, perm)), boot=0)[0])
    say(f"  {'offsets shuffled across authors':<34} sleep OR mean {np.mean(sh):.3f} "
        f"(range {min(sh):.3f} to {max(sh):.3f}, 20 shuffles)")
    out["shuffled"] = sh
    agree = np.mean([community[a] == ind[a] for a in elig])
    say(f"  individual offset equals community offset for {100 * agree:.0f}% of authors")
    res["T2_2"] = out

    with open(os.path.join(AV.OUT_DIR, "revision3_checks.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=lambda x: x if not isinstance(x, np.generic) else x.item())
    print("\n" + "=" * 70 + "\nSUMMARY FOR REVISION 3 (paste this into the chat)\n" + "=" * 70)
    print("\n".join(OUT))
    print("=" * 70)


if __name__ == "__main__":
    main()
