#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPST-FoMO: four-space feature extraction
========================================

Turns the item-level corpus into ONE ROW PER AUTHOR-WINDOW, with four feature
blocks that are kept strictly separate so the leave-one-space-out ablation is
meaningful.

    C_*   Cyber      community portfolio, affordance use, reception
    P_*   Physical   temporal trace, estimated circadian position, bursts
    S_*   Social     reply structure, reciprocity, interaction partners
    T_*   Thinking   lexical and stylistic markers in the text

Applied at extraction, all of them reportable:

  window filter     records outside 2023-01-01 to 2025-12-31 are excluded.
                    The collector let some 2026 thread replies through; they
                    stay in the database and are dropped here.
  cohort filter     seed and control only. Participants are the reserve pool.
  volume floor      authors below MIN_RECORDS are dropped, because per-author
                    circadian estimation is unreliable below roughly 30 records.
  rematching        --rematch redoes seed/control matching on POST-EXPANSION
                    volume and tenure. The original matching was done before
                    expansion and no longer holds: controls were drawn from
                    thread commenters, who are systematically heavier users.

Two passes, to keep memory bounded on a 1M-record corpus: metadata first for
C, P and S, then a streaming pass over the text for T.

Usage
    python extract_cpst_features.py
    python extract_cpst_features.py --rematch --min-records 30
    python extract_cpst_features.py --out features_v1.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Data folder: set CPST_BASE_DIR to override the default location.
BASE_DIR = os.environ.get(
    "CPST_BASE_DIR",
    os.path.join(os.path.expanduser("~"), "Documents", "CPST_FOMO_v6"))
DB_PATH = os.path.join(BASE_DIR, "cpst_fomo.db")

WINDOW_START = "2023-01-01"
WINDOW_END = "2025-12-31"
MIN_RECORDS = 30
ANCHOR_LOCAL_HOUR = 4          # must be an integer, see estimate_utc_offset
SESSION_GAP_S = 30 * 60          # 30 minutes defines a session boundary
BURST_S = 5 * 60                 # inter-arrival below this counts as a burst
STUDY_BLOCKS = ("anxiety", "disengagement")

WINDOW_START_TS = datetime.fromisoformat(WINDOW_START).replace(
    tzinfo=timezone.utc).timestamp()
WINDOW_END_TS = datetime.fromisoformat(WINDOW_END).replace(
    tzinfo=timezone.utc).timestamp()

SOCIAL_DOMAINS = ("instagram", "tiktok", "twitter", "x.com", "facebook",
                  "snapchat", "youtube", "discord", "twitch", "linkedin")


# ======================================================================
# T-space lexicons
# ======================================================================
# Small, transparent and auditable. Published as supplementary material.
# Deliberately NOT a sentiment model: the point is interpretable features
# that can be defended one by one, not a black box inside a feature block.

LEX = {
    "self_sg":     r"\b(i|me|my|mine|myself|i'm|i've|i'd|i'll)\b",
    "self_pl":     r"\b(we|us|our|ours|ourselves)\b",
    "other_ref":   r"\b(they|them|their|everyone|everybody|people|friends|"
                   r"others|someone|anyone|others')\b",
    "second":      r"\b(you|your|yours|yourself)\b",
    "negation":    r"\b(not|no|never|none|nothing|nobody|cannot|can't|won't|"
                   r"don't|doesn't|didn't|isn't|aren't|wasn't)\b",
    "absolutist":  r"\b(always|never|completely|totally|entirely|absolutely|"
                   r"constantly|everything|nothing|everyone|nobody|all the time)\b",
    "comparative": r"\b(\w+er than|more than|less than|better than|worse than|"
                   r"compared to|comparing|as much as|instead of)\b",
    "past":        r"\b(was|were|had|did|used to|ago|yesterday|last night|"
                   r"last week|back then|should have|could have|would have)\b",
    "future":      r"\b(will|going to|gonna|soon|tomorrow|next week|about to|"
                   r"planning|upcoming|later)\b",
    "uncertainty": r"\b(maybe|perhaps|probably|might|guess|sort of|kind of|"
                   r"i think|not sure|somehow)\b",
    "exclusion":   r"\b(left out|excluded|without me|not invited|didn't invite|"
                   r"behind|missing out|missed out|out of the loop)\b",
    "compulsion":  r"\b(keep checking|can't stop|refresh|scrolling|again and "
                   r"again|every few minutes|compulsive|constantly checking)\b",
    "somatic":     r"\b(heart racing|chest|breathe|breathing|nausea|nauseous|"
                   r"sick to my stomach|shaking|dizzy|exhausted|headache)\b",
    "sleep":       r"\b(sleep|asleep|insomnia|awake|3am|4am|2am|all night|"
                   r"couldn't sleep|stayed up|tired)\b",
    "affect_neg":  r"\b(anxious|anxiety|sad|depressed|lonely|miserable|awful|"
                   r"terrible|hate|scared|afraid|worried|upset|hurt|guilty|"
                   r"ashamed|jealous|envious|angry|frustrated)\b",
    "affect_pos":  r"\b(happy|glad|grateful|proud|calm|relieved|better|good|"
                   r"enjoy|enjoyed|love|excited|hopeful)\b",
}
LEX_RE = {k: re.compile(v, re.IGNORECASE) for k, v in LEX.items()}
TOKEN_RE = re.compile(r"[a-z']+", re.IGNORECASE)


# ======================================================================
# helpers
# ======================================================================

def shannon(counts) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    ps = [c / total for c in counts if c > 0]
    return -sum(p * math.log(p, 2) for p in ps)


def normalised_entropy(counts) -> float:
    k = sum(1 for c in counts if c > 0)
    if k <= 1:
        return 0.0
    return shannon(counts) / math.log(k, 2)


def estimate_utc_offset(hours_utc: np.ndarray) -> Tuple[int, float]:
    """Estimate an author's UTC offset from their own activity trough.

    Assumption: the 6-hour window with least posting is the sleep window, and it
    is centred at 04:00 local. Della Negra et al. (arXiv 2605.04371) report that
    the activity minimum falls consistently close to 04:00 local across
    virtually all time zones on this platform.

    The anchor must be an INTEGER. With a half-integer anchor such as 03:30,
    (anchor - trough_centre) always ends in .5 because trough_centre is an
    integer, and Python rounds halves to even, so only EVEN offsets can ever be
    produced and every odd time zone is folded into a neighbour. An integer
    anchor removes that degeneracy entirely.

    This is a per-author estimate with real error, and it is why the paper must
    describe the Physical space as an inferred proxy rather than a measurement.

    Returns (offset_hours, trough_depth). trough_depth is the ratio of trough
    density to overall mean density; values near 1 mean no detectable rhythm
    and the offset should be treated as missing.
    """
    counts = np.bincount(hours_utc, minlength=24).astype(float)
    if counts.sum() == 0:
        return 0, 1.0
    wrapped = np.concatenate([counts, counts[:6]])
    win = np.array([wrapped[i:i + 6].sum() for i in range(24)])
    trough_start = int(np.argmin(win))
    trough_centre = (trough_start + 3) % 24
    offset = int(round((ANCHOR_LOCAL_HOUR - trough_centre) % 24))
    if offset > 12:
        offset -= 24
    mean_6h = counts.sum() / 4.0
    depth = float(win[trough_start] / mean_6h) if mean_6h > 0 else 1.0
    return offset, depth


def circular_resultant(hours_local: np.ndarray) -> float:
    """Vector strength of the daily rhythm: 0 is uniform, 1 is a single hour."""
    if len(hours_local) == 0:
        return 0.0
    theta = 2 * np.pi * (hours_local % 24) / 24.0
    return float(np.hypot(np.cos(theta).mean(), np.sin(theta).mean()))


# ======================================================================
# pass 1: metadata -> C, P, S
# ======================================================================

def apply_half(df: pd.DataFrame, half: int) -> pd.DataFrame:
    """Keep alternating records per author, for split-half reliability.

    Records are ordered by time within author and assigned alternately, so each
    half spans the whole observation window rather than the first or second part
    of it. A chronological split would confound reliability with change over
    time; an alternating split does not.
    """
    df = df.sort_values(["author_hash", "created_utc"]).copy()
    df["_rank"] = df.groupby("author_hash").cumcount()
    keep = df["_rank"] % 2 == (0 if half == 1 else 1)
    out = df[keep].drop(columns=["_rank"])
    print(f"  half {half}: {len(out):,} of {len(df):,} records retained")
    return out


def load_metadata(conn: sqlite3.Connection, min_records: int) -> pd.DataFrame:
    blocks = ",".join("?" * 2)
    sql = f"""
        SELECT i.id, i.kind, i.subreddit, i.block, i.author_hash, i.created_utc,
               i.n_chars, i.score, i.upvote_ratio, i.parent_id, i.link_id,
               i.depth, i.is_self, i.domain, i.edited,
               a.cohort, a.seed_block
        FROM items i
        JOIN authors a ON a.author_hash = i.author_hash
        WHERE i.author_hash IS NOT NULL
          AND a.cohort IN ('seed','control')
          AND i.created_utc BETWEEN ? AND ?
    """
    df = pd.read_sql_query(sql, conn, params=(WINDOW_START_TS, WINDOW_END_TS))
    print(f"  in-window records for seed/control authors: {len(df):,}")

    vc = df.author_hash.value_counts()
    keep = set(vc[vc >= min_records].index)
    df = df[df.author_hash.isin(keep)].copy()
    print(f"  authors at or above {min_records} records: {len(keep):,}")
    print(f"  records retained: {len(df):,}")
    return df


def cyber_physical_social(df: pd.DataFrame) -> pd.DataFrame:
    # id -> (author, ts) for parent lookups
    id_author = dict(zip(df.id, df.author_hash))
    id_ts = dict(zip(df.id, df.created_utc))

    df = df.sort_values(["author_hash", "created_utc"])
    df["parent_bare"] = df.parent_id.str.slice(3)
    df["parent_author"] = df.parent_bare.map(id_author)
    df["parent_ts"] = df.parent_bare.map(id_ts)

    # who replied to whom, for in-degree
    in_degree = Counter(df.parent_author.dropna())

    rows = []
    for author, g in df.groupby("author_hash", sort=False):
        ts = g.created_utc.to_numpy()
        n = len(g)
        hours_utc = ((ts // 3600) % 24).astype(int)

        # ---------------- Cyber ----------------
        sub_counts = g.subreddit.value_counts()
        block_counts = g.block.value_counts()
        n_days = max((ts.max() - ts.min()) / 86400.0, 1.0)

        half = ts.min() + (ts.max() - ts.min()) / 2
        subs_first = set(g.loc[g.created_utc <= half, "subreddit"])
        subs_second = set(g.loc[g.created_utc > half, "subreddit"])
        union = subs_first | subs_second
        jaccard = (len(subs_first & subs_second) / len(union)) if union else 0.0

        domains = g.domain.dropna()
        social_links = sum(1 for d in domains
                           if any(s in str(d).lower() for s in SOCIAL_DOMAINS))

        C = {
            "C_n_communities":     int(sub_counts.size),
            "C_community_entropy": normalised_entropy(sub_counts.values),
            "C_top_community_share": float(sub_counts.iloc[0] / n),
            "C_anxiety_share":     float(block_counts.get("anxiety", 0) / n),
            "C_diseng_share":      float(block_counts.get("disengagement", 0) / n),
            "C_offblock_share":    float(block_counts.get("other", 0) / n),
            "C_post_share":        float((g.kind == "post").mean()),
            "C_median_score":      float(g.score.median()),
            "C_neg_score_share":   float((g.score < 1).mean()),
            "C_selfpost_share":    float(g.is_self.fillna(0).mean()),
            "C_n_link_domains":    int(domains.nunique()),
            "C_social_link_share": float(social_links / max(len(domains), 1)),
            "C_edited_share":      float(g.edited.notna().mean()),
            "C_community_churn":   1.0 - jaccard,
            "C_communities_per_month": float(sub_counts.size / (n_days / 30.0)),
        }

        # ---------------- Physical ----------------
        offset, trough_depth = estimate_utc_offset(hours_utc)
        hours_local = (hours_utc + offset) % 24
        gaps = np.diff(ts)
        active_days = len(set((ts // 86400).astype(int)))
        sessions = 1 + int((gaps > SESSION_GAP_S).sum()) if n > 1 else 1
        dow = ((ts // 86400 + 4) % 7).astype(int)   # epoch day 0 was a Thursday

        # latency of this author's replies to other people
        lat = (g.created_utc - g.parent_ts)
        lat = lat[(g.parent_author.notna()) & (g.parent_author != author)]
        lat = lat[(lat > 0) & (lat < 7 * 86400)]

        P = {
            "P_utc_offset_est":    offset,
            "P_rhythm_trough_depth": trough_depth,
            "P_rhythm_strength":   circular_resultant(hours_local),
            "P_hour_entropy":      normalised_entropy(np.bincount(hours_local, minlength=24)),
            "P_night_share":       float(((hours_local >= 0) & (hours_local < 5)).mean()),
            "P_latenight_share":   float(((hours_local >= 1) & (hours_local < 4)).mean()),
            "P_workhours_share":   float(((hours_local >= 9) & (hours_local < 18)).mean()),
            "P_weekend_share":     float(np.isin(dow, [5, 6]).mean()),
            "P_median_gap_s":      float(np.median(gaps)) if n > 1 else np.nan,
            "P_burst_share":       float((gaps < BURST_S).mean()) if n > 1 else 0.0,
            "P_n_sessions":        sessions,
            "P_items_per_session": float(n / sessions),
            "P_active_days":       active_days,
            "P_activity_density":  float(active_days / n_days),
            "P_items_per_active_day": float(n / active_days),
            "P_reply_latency_med": float(lat.median()) if len(lat) else np.nan,
            "P_fast_reply_share":  float((lat < 300).mean()) if len(lat) else np.nan,
        }

        # ---------------- Social ----------------
        is_comment = g.kind == "comment"
        replies_to_comment = g.parent_id.fillna("").str.startswith("t1_")
        partners = g.parent_author.dropna()
        partners = partners[partners != author]
        mutual = 0
        partner_set = set(partners)
        if partner_set:
            replied_back = set(
                df.loc[df.parent_author == author, "author_hash"])
            mutual = len(partner_set & replied_back)

        S = {
            "S_comment_share":     float(is_comment.mean()),
            "S_reply_to_comment_share": float(replies_to_comment.mean()),
            "S_mean_depth":        float(g.depth.dropna().mean()) if g.depth.notna().any() else np.nan,
            "S_max_depth":         float(g.depth.dropna().max()) if g.depth.notna().any() else np.nan,
            "S_n_partners":        int(len(partner_set)),
            "S_partners_per_item": float(len(partner_set) / n),
            "S_replies_received":  int(in_degree.get(author, 0)),
            "S_reply_ratio":       float(in_degree.get(author, 0) / n),
            "S_mutual_partners":   mutual,
            "S_reciprocity":       float(mutual / max(len(partner_set), 1)),
            "S_n_threads":         int(g.link_id.nunique()),
            "S_items_per_thread":  float(n / max(g.link_id.nunique(), 1)),
        }

        rows.append({
            "author_hash": author,
            "cohort": g.cohort.iloc[0],
            "seed_block": g.seed_block.iloc[0],
            "n_items": n,
            "tenure_days": n_days,
            "first_seen": float(ts.min()),
            "last_seen": float(ts.max()),
            **C, **P, **S,
        })

    return pd.DataFrame(rows)


# ======================================================================
# pass 2: streaming text -> T
# ======================================================================

def thinking_features(conn: sqlite3.Connection, authors: set,
                      keep_ids: Optional[set] = None) -> pd.DataFrame:
    acc: Dict[str, Counter] = defaultdict(Counter)
    tokens: Dict[str, int] = defaultdict(int)
    records: Dict[str, int] = defaultdict(int)
    lengths: Dict[str, List[int]] = defaultdict(list)
    vocab: Dict[str, set] = defaultdict(set)

    cur = conn.execute("""
        SELECT id, author_hash, body_raw
        FROM items
        WHERE author_hash IS NOT NULL
          AND created_utc BETWEEN ? AND ?
          AND body_raw IS NOT NULL
    """, (WINDOW_START_TS, WINDOW_END_TS))

    seen = 0
    while True:
        chunk = cur.fetchmany(20000)
        if not chunk:
            break
        for item_id, author, text in chunk:
            if author not in authors or not text:
                continue
            if keep_ids is not None and item_id not in keep_ids:
                continue
            toks = TOKEN_RE.findall(text.lower())
            if not toks:
                continue
            n_tok = len(toks)
            tokens[author] += n_tok
            records[author] += 1
            lengths[author].append(n_tok)
            if len(vocab[author]) < 20000:
                vocab[author].update(toks)
            c = acc[author]
            for name, rx in LEX_RE.items():
                hits = len(rx.findall(text))
                if hits:
                    c[name] += hits
            c["q_marks"] += text.count("?")
            c["excl"] += text.count("!")
            c["ellipsis"] += text.count("...")
        seen += len(chunk)
        if seen % 200000 == 0:
            print(f"    text pass: {seen:,} records")

    rows = []
    for author in authors:
        n_tok = tokens.get(author, 0)
        if n_tok == 0:
            continue
        c = acc[author]
        per_k = lambda k: 1000.0 * c.get(k, 0) / n_tok
        ln = lengths[author]
        rows.append({
            "author_hash": author,
            "T_tokens_total":     n_tok,
            "T_mean_tokens":      float(np.mean(ln)),
            "T_median_tokens":    float(np.median(ln)),
            "T_len_variability":  float(np.std(ln) / max(np.mean(ln), 1)),
            "T_type_token_ratio": len(vocab[author]) / n_tok,
            "T_self_sg":          per_k("self_sg"),
            "T_self_pl":          per_k("self_pl"),
            "T_other_ref":        per_k("other_ref"),
            "T_second_person":    per_k("second"),
            "T_self_other_ratio": c.get("self_sg", 0) / max(c.get("other_ref", 0), 1),
            "T_negation":         per_k("negation"),
            "T_absolutist":       per_k("absolutist"),
            "T_comparative":      per_k("comparative"),
            "T_past":             per_k("past"),
            "T_future":           per_k("future"),
            "T_temporal_balance": (c.get("future", 0) - c.get("past", 0)) /
                                  max(c.get("future", 0) + c.get("past", 0), 1),
            "T_uncertainty":      per_k("uncertainty"),
            "T_exclusion_lex":    per_k("exclusion"),
            "T_compulsion_lex":   per_k("compulsion"),
            "T_somatic_lex":      per_k("somatic"),
            "T_sleep_lex":        per_k("sleep"),
            "T_affect_neg":       per_k("affect_neg"),
            "T_affect_pos":       per_k("affect_pos"),
            "T_affect_balance":   (c.get("affect_pos", 0) - c.get("affect_neg", 0)) /
                                  max(c.get("affect_pos", 0) + c.get("affect_neg", 0), 1),
            "T_question_rate":    per_k("q_marks"),
            "T_exclaim_rate":     per_k("excl"),
            "T_ellipsis_rate":    per_k("ellipsis"),
        })
    return pd.DataFrame(rows)


# ======================================================================
# rematching
# ======================================================================

def rematch(df: pd.DataFrame) -> pd.DataFrame:
    """Nearest-neighbour match on post-expansion volume and tenure, within block.

    The original matching used pre-expansion counts and no longer holds:
    controls were drawn from thread commenters, who post more. Without this,
    any seed/control difference is confounded with activity level.
    """
    df = df.copy()
    df["matched"] = 0
    df["pair_id"] = -1          # shared by the seed and control of each pair
    df["_lv"] = np.log1p(df.n_items)
    df["_lt"] = np.log1p(df.tenure_days)
    for col in ("_lv", "_lt"):
        s = df[col].std()
        df[col] = (df[col] - df[col].mean()) / (s if s > 0 else 1)

    used = set()
    pairs = 0
    for block, g in df.groupby("seed_block"):
        seeds = g[g.cohort == "seed"]
        ctrls = g[g.cohort == "control"].copy()
        if seeds.empty or ctrls.empty:
            continue
        cx = ctrls[["_lv", "_lt"]].to_numpy()
        cidx = ctrls.index.to_numpy()
        for si, srow in seeds.iterrows():
            sv = np.array([srow._lv, srow._lt])
            d = np.linalg.norm(cx - sv, axis=1)
            order = np.argsort(d)
            for j in order:
                ci = cidx[j]
                if ci in used:
                    continue
                if d[j] > 0.5:          # caliper, in pooled SD units
                    break
                used.add(ci)
                df.loc[si, "matched"] = 1
                df.loc[ci, "matched"] = 1
                df.loc[si, "pair_id"] = pairs
                df.loc[ci, "pair_id"] = pairs
                pairs += 1
                break

    df = df.drop(columns=["_lv", "_lt"])
    print(f"  matched pairs: {pairs} ({2 * pairs} authors retained as matched)")
    unmatched = (df.matched == 0).sum()
    print(f"  unmatched authors: {unmatched} (kept, flagged matched=0)")
    return df


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
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--min-records", type=int, default=MIN_RECORDS)
    ap.add_argument("--rematch", action="store_true")
    ap.add_argument("--half", type=int, choices=[1, 2], default=None,
                    help="split-half reliability: keep alternating records per "
                         "author. Run once with 1 and once with 2.")
    ap.add_argument("--out", default="cpst_features.csv")
    args = _args(ap)

    if not os.path.exists(args.db):
        sys.exit(f"Database not found: {args.db}")

    conn = sqlite3.connect(args.db)

    print("pass 1: metadata")
    meta = load_metadata(conn, args.min_records)
    if meta.empty:
        sys.exit("No records survived the filters.")

    keep_ids = None
    if args.half:
        print(f"split-half mode, half {args.half}")
        meta = apply_half(meta, args.half)
        keep_ids = set(meta.id)
        vc = meta.author_hash.value_counts()
        # the volume floor applies to the half, not the whole
        meta = meta[meta.author_hash.isin(vc[vc >= max(10, args.min_records // 2)].index)]
        keep_ids = set(meta.id)
        print(f"  authors retained in this half: {meta.author_hash.nunique():,}")

    print("computing C, P, S features")
    feats = cyber_physical_social(meta)
    print(f"  {len(feats):,} author-windows")

    print("pass 2: text")
    tf = thinking_features(conn, set(feats.author_hash), keep_ids)
    print(f"  {len(tf):,} author-windows with text")

    out = feats.merge(tf, on="author_hash", how="left")

    if args.rematch:
        print("rematching cohorts on post-expansion volume and tenure")
        out = rematch(out)

    conn.close()

    out_name = args.out
    if args.half and out_name == "cpst_features.csv":
        out_name = f"cpst_features_h{args.half}.csv"
    path = os.path.join(BASE_DIR, out_name)
    out.to_csv(path, index=False)

    blocks = {"C": [c for c in out.columns if c.startswith("C_")],
              "P": [c for c in out.columns if c.startswith("P_")],
              "S": [c for c in out.columns if c.startswith("S_")],
              "T": [c for c in out.columns if c.startswith("T_")]}

    meta_path = os.path.join(BASE_DIR, "feature_manifest.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated": datetime.now(timezone.utc).isoformat(),
            "window": [WINDOW_START, WINDOW_END],
            "min_records": args.min_records,
            "rematched": bool(args.rematch),
            "n_author_windows": len(out),
            "cohorts": out.cohort.value_counts().to_dict(),
            "blocks": {k: v for k, v in blocks.items()},
            "block_sizes": {k: len(v) for k, v in blocks.items()},
            "lexicons": LEX,
            "session_gap_seconds": SESSION_GAP_S,
            "burst_threshold_seconds": BURST_S,
        }, f, indent=2)

    print("\n" + "=" * 62)
    print(f"author-windows : {len(out):,}")
    for k, v in blocks.items():
        print(f"  {k} features   : {len(v)}")
    print(f"cohorts        : {out.cohort.value_counts().to_dict()}")
    if args.rematch:
        print(f"matched        : {int(out.matched.sum()):,}")
    print(f"\nwritten to     : {path}")
    print(f"manifest       : {meta_path}")
    print("=" * 62)

    # quick sanity view
    print("\nmissingness above 5%:")
    miss = out.isna().mean().sort_values(ascending=False)
    for col, m in miss[miss > 0.05].items():
        print(f"  {col:<28}{100*m:>6.1f}%")


if __name__ == "__main__":
    main()
