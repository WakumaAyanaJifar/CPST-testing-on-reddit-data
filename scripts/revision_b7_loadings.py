#!/usr/bin/env python3
"""
B7: sampling stability of canonical structure coefficients (reviewer point 5).

For each pair of spaces, in the conservative specification used for Table 5,
the first canonical variates are estimated on the full sample and the four
largest structure coefficients on each side are taken. The same quantities are
then re-estimated on 500 random halves of the authors (drawn without
replacement, as in revision_analyses.py). Because the sign of a canonical
variate is arbitrary, each half-sample variate is aligned to the full-sample
one before its coefficients are computed. Intervals are the estimate plus or
minus 1.96 half-sample standard deviations.

Put it in the same scripts folder and run:
    python scripts\\revision_b7_loadings.py
"""
from __future__ import annotations

import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_validity as AV          # noqa: E402
import analyze_spaces_v2 as AS2        # noqa: E402
import revision_analyses as RA         # noqa: E402

N_HALF, TOP = 500, 4


def variates(X, ca, cb):
    sub = X[ca + cb].dropna()
    Pa, _ = AS2.pcs_to(sub[ca].to_numpy(), 0.8)
    Pb, _ = AS2.pcs_to(sub[cb].to_numpy(), 0.8)
    rho, u, v = AS2.cca_variates(Pa, Pb)
    return sub, rho, u, v


def coefs(sub, cols, var):
    return {c: float(pd.Series(var).corr(sub[c].reset_index(drop=True), method="spearman"))
            for c in cols}


def main():
    df = pd.read_csv(os.path.join(AV.BASE_DIR, "cpst_features.csv"))
    cols = [c for c in df.columns if c not in AV.NON_FEATURES
            and AV.block_of(c) in AV.BLOCKS and c not in AV.DROP_FEATURES]
    cons = [c for c in cols if c not in AS2.COUNT_SCALING and c not in AS2.PARTIAL_COVERAGE_FEATURES]
    by = {b: [c for c in cons if AV.block_of(c) == b] for b in AV.BLOCKS}
    X, _ = RA.prep(df, cols)
    rng = np.random.default_rng(42)
    n = len(df)
    halves = [rng.choice(n, n // 2, replace=False) for _ in range(N_HALF)]

    print("Structure coefficients, conservative specification, "
          f"{N_HALF} half-samples\n")
    for a, b in combinations(AV.BLOCKS, 2):
        sub, rho, u, v = variates(X, by[a], by[b])
        full_a, full_b = coefs(sub, by[a], u), coefs(sub, by[b], v)
        top_a = sorted(full_a, key=lambda c: -abs(full_a[c]))[:TOP]
        top_b = sorted(full_b, key=lambda c: -abs(full_b[c]))[:TOP]
        draws = {c: [] for c in top_a + top_b}
        for idx in halves:
            Xh, _ = RA.prep(df.iloc[idx].reset_index(drop=True), cols)
            sh, _, uh, vh = variates(Xh, by[a], by[b])
            # align the arbitrary sign of each variate with the full-sample one
            ca, cb = coefs(sh, top_a, uh), coefs(sh, top_b, vh)
            sa = np.sign(sum(ca[c] * full_a[c] for c in top_a)) or 1.0
            sb = np.sign(sum(cb[c] * full_b[c] for c in top_b)) or 1.0
            for c in top_a:
                draws[c].append(sa * ca[c])
            for c in top_b:
                draws[c].append(sb * cb[c])
        print(f"{a}-{b}   rho1 {rho:.3f}")
        for side, top, full in ((a, top_a, full_a), (b, top_b, full_b)):
            for c in top:
                sd = float(np.nanstd(draws[c], ddof=1))
                lo, hi = full[c] - 1.96 * sd, full[c] + 1.96 * sd
                stable = "sign stable" if lo * hi > 0 else "SIGN NOT STABLE"
                print(f"   {side}  {c:<28}{full[c]:+.2f}  [{lo:+.2f}, {hi:+.2f}]  {stable}")
        print()
    print("Paste everything above into the chat.")


if __name__ == "__main__":
    main()
