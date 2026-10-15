#!/usr/bin/env python3
"""
B1 follow-up: physical sub-group coherence WITHOUT definitionally linked pairs.

Some physical features are arithmetic functions of each other once record count
and tenure are residualized out, so their high correlation is partly built in:
    active days, activity density (= active days / tenure),
    records per active day (= records / active days)        -> density group
    sessions, records per session (= records / sessions)      -> tempo group
    night share (00-05) and late-night share (01-04, nested)  -> rhythm group
This script drops those pairs from BOTH the within- and between-group sets and
re-tests the grouping against random groupings of equal size.

Put it in the same scripts folder as revision_analyses.py and run:
    python scripts\\revision_b1_check.py
"""
from __future__ import annotations

import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_validity as AV          # noqa: E402
import revision_analyses as RA         # noqa: E402

LINKED = {frozenset(p) for p in [
    ("P_active_days", "P_activity_density"),
    ("P_active_days", "P_items_per_active_day"),
    ("P_activity_density", "P_items_per_active_day"),
    ("P_n_sessions", "P_items_per_session"),
    ("P_night_share", "P_latenight_share"),
]}
N_PERM = 2000


def margin(corr, lab, cols):
    w, b = [], []
    for i, j in combinations(cols, 2):
        if frozenset((i, j)) in LINKED:
            continue
        r = abs(corr.at[i, j])
        if np.isfinite(r):
            (w if lab[i] == lab[j] else b).append(r)
    return (np.median(w) - np.median(b)) if w and b else np.nan, w, b


def main():
    df = pd.read_csv(os.path.join(AV.BASE_DIR, "cpst_features.csv"))
    cols = [c for c in df.columns if c not in AV.NON_FEATURES
            and AV.block_of(c) in AV.BLOCKS and c not in AV.DROP_FEATURES]
    _, corr = RA.prep(df, cols)
    groups = {g: [c for c in fs if c in cols] for g, fs in RA.PHYS_GROUPS.items()}
    phys = [c for fs in groups.values() for c in fs]
    lab = {c: g for g, fs in groups.items() for c in fs}

    print("Physical sub-groups, definitionally linked pairs excluded\n")
    print(f"  {'group':<9}{'pairs left':>11}{'within':>9}{'between':>9}{'gap':>8}")
    for g, fs in groups.items():
        w = [abs(corr.at[i, j]) for i, j in combinations(fs, 2)
             if frozenset((i, j)) not in LINKED and np.isfinite(corr.at[i, j])]
        b = [abs(corr.at[i, j]) for i in fs for j in phys
             if lab[j] != g and np.isfinite(corr.at[i, j])]
        if w:
            print(f"  {g:<9}{len(w):>11}{np.median(w):>9.3f}{np.median(b):>9.3f}"
                  f"{np.median(w) - np.median(b):>+8.3f}")
        else:
            print(f"  {g:<9}{0:>11}   no pairs left: the group is one quantity")

    obs, _, _ = margin(corr, lab, phys)
    rng = np.random.default_rng(42)
    labs = np.array([lab[c] for c in phys], dtype=object)
    null = np.array([margin(corr, dict(zip(phys, rng.permutation(labs))), phys)[0]
                     for _ in range(N_PERM)])
    p = (np.sum(null >= obs) + 1) / (N_PERM + 1)
    print(f"\n  grouping margin {obs:+.4f}   random p95 {np.nanpercentile(null, 95):+.4f}"
          f"   p = {p:.4f}   ({N_PERM} permutations)")
    print("\n  Paste everything above into the chat.")


if __name__ == "__main__":
    main()
