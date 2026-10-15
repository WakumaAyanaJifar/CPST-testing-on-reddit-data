#!/usr/bin/env python3
"""
Power of decision rule R4 (CPST specificity) by simulation.

Synthetic authors (n = 2595) have 64 features that combine a general factor, a
trait factor for their CPST space, and a method factor for their data channel,
at the observed correlation levels. The trait share of the structured variance
is varied from 0 to 1; for each setting, 200 datasets are simulated and R4 is
applied exactly as in the paper. Needs no private data.

    python scripts/r4_power.py
"""
import numpy as np, pandas as pd, sys
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_validity as AV
from revision2_channel import channel_of
# feature list exactly as in the paper
import re
names = sorted(set(re.findall(r'`([CPST]_[a-z0-9_]+)`', open(os.path.join(HERE, '..', 'data', 'DATA_DICTIONARY.md'), encoding='utf-8').read())))
cols = [c for c in names if c not in AV.DROP_FEATURES]
assert len(cols) == 64, len(cols)
cpst = np.array([AV.block_of(c) for c in cols]); chan = np.array([channel_of(c) for c in cols])
cmap = {"metadata":"C","time":"P","thread":"S","content":"T"}
diff = np.where(cpst != np.array([cmap[x] for x in chan]))[0]
iu = np.triu_indices(64, 1)

def margin(R, lab):
    w = lab[iu[0]] == lab[iu[1]]
    v = R[iu]
    return np.median(v[w]) - np.median(v[~w])

def closest_ok(R, f):
    # is feature f most associated with its CPST block (others in that block, excluding cross-channel features)?
    best, bv = None, -1
    for b in "CPST":
        idx = [j for j in range(64) if cpst[j] == b and j not in diff and j != f]
        m = np.median(R[f, idx])
        if m > bv: best, bv = b, m
    return best == cpst[f]

def one(rng, tau, n=2595, g2=0.05, s2=0.03, nperm=200):
    a, b = np.sqrt(tau * s2), np.sqrt((1 - tau) * s2)
    T = {k: rng.standard_normal(n) for k in "CPST"}
    M = {k: rng.standard_normal(n) for k in sorted(set(chan))}
    G = rng.standard_normal(n)
    X = np.empty((n, 64))
    for j in range(64):
        X[:, j] = np.sqrt(g2) * G + a * T[cpst[j]] + b * M[chan[j]] + np.sqrt(1 - g2 - s2) * rng.standard_normal(n)
    R = np.abs(np.corrcoef(X, rowvar=False))
    m_c, m_ch = margin(R, cpst), margin(R, np.array([cmap[x] for x in chan]))
    null = []
    for _ in range(nperm):
        lab = cpst.copy()
        for ch in sorted(set(chan)):
            idx = np.where(chan == ch)[0]
            lab[idx] = rng.permutation(lab[idx])
        null.append(margin(R, lab))
    p95 = np.percentile(null, 95)
    feats = all(closest_ok(R, f) for f in diff)
    return (m_c > m_ch) and (m_c > p95) and feats, m_c > m_ch, m_c > p95, feats, m_c, m_ch

rng = np.random.default_rng(7)
print("trait share | R4 passes | CPST>channel | CPST>null95 | 4 features follow CPST | mean margins CPST/channel")
for tau in [0.0, 0.25, 0.5, 0.75, 1.0]:
    res = [one(rng, tau) for _ in range(200)]
    a = np.array([r[:4] for r in res], float); mc = np.mean([r[4] for r in res]); mh = np.mean([r[5] for r in res])
    print(f"   {tau:4.2f}     |  {a[:,0].mean():5.2f}    |   {a[:,1].mean():5.2f}      |   {a[:,2].mean():5.2f}     |   {a[:,3].mean():5.2f}     | {mc:+.4f} / {mh:+.4f}")
