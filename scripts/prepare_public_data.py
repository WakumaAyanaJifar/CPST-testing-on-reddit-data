#!/usr/bin/env python3
"""
Prepare the author-level feature files for public release.

Run this ONCE, on the machine that holds the private files, and commit only
what it writes. It reads

    <CPST_BASE_DIR>/cpst_features.csv
    <CPST_BASE_DIR>/cpst_features_h1.csv
    <CPST_BASE_DIR>/cpst_features_h2.csv

and writes de-identified copies to ./data (or --out).

What it does, and why
---------------------
1. Replaces each salted author hash with a fresh random identifier (a00001,
   a00002, ...). The column keeps the name author_hash so the analysis
   scripts run unchanged. The same author gets the same identifier in all three files,
   so split-half reliability can still be computed, but the identifiers cannot
   be linked to the private database. The mapping is never written to disk.
2. Removes exact timestamps (first_seen, last_seen and any other column that
   holds Unix times). An author's first and last posting time, together with
   their record count, could be matched against public Reddit archives.
3. Keeps row order and matched-pair numbers exactly as in the private files.
   Permutation tests and cross-validation folds are seeded but depend on row
   order and pair labels, so changing either would make the published numbers
   impossible to reproduce. Neither reveals anything: rows are ordered by the
   private salted hash, which is random without the salt, and pair numbers
   only record the order in which pairs were formed.
4. Refuses to write anything if a 32-character hexadecimal string (the form of
   the private hashes) survives in any cell.

Everything the published analyses need is kept: all 71 features, record count,
tenure, cohort, community block, matched flag and pair identifier.

Usage
-----
    python scripts/prepare_public_data.py
    python scripts/prepare_public_data.py --out data
"""
from __future__ import annotations

import argparse
import os
import re
import secrets
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.environ.get(
    "CPST_BASE_DIR",
    os.path.join(os.path.expanduser("~"), "Documents", "CPST_FOMO_v6"))
FILES = ["cpst_features.csv", "cpst_features_h1.csv", "cpst_features_h2.csv"]

ALWAYS_DROP = {"first_seen", "last_seen"}
UNIX_FLOOR = 1.0e9            # 2001-09-09; any column whose values sit above
                              # this is treated as a timestamp and removed
HEX32 = re.compile(r"\b[0-9a-f]{32}\b")


def timestamp_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in ALWAYS_DROP:
            cols.append(c)
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(v) and v.median() > UNIX_FLOOR:
                cols.append(c)
    return cols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=BASE_DIR)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    frames = {}
    for name in FILES:
        path = os.path.join(args.src, name)
        if not os.path.exists(path):
            sys.exit(f"missing: {path}")
        frames[name] = pd.read_csv(path)
        print(f"read {name}: {len(frames[name]):,} rows, "
              f"{frames[name].shape[1]} columns")

    # one random identifier per author, shared across the three files
    authors = sorted(set().union(*(set(f.author_hash) for f in frames.values())))
    rng = secrets.SystemRandom()
    order = list(range(1, len(authors) + 1))
    rng.shuffle(order)
    new_id = {a: f"a{n:05d}" for a, n in zip(authors, order)}

    os.makedirs(args.out, exist_ok=True)
    for name, df in frames.items():
        drop = timestamp_columns(df)
        out = df.drop(columns=drop)
        # the column keeps its name so the analysis scripts run unchanged,
        # but it now holds a random identifier, not a hash
        out["author_hash"] = out.author_hash.map(new_id)

        leaked = [c for c in out.columns
                  if out[c].astype(str).str.contains(HEX32).any()]
        if leaked:
            sys.exit(f"ABORT: hash-like strings remain in {name}: {leaked}")

        dest = os.path.join(args.out, name)
        out.to_csv(dest, index=False)
        print(f"wrote {dest}: {len(out):,} rows, {out.shape[1]} columns; "
              f"removed {drop}")

    print("\nThe author mapping was held in memory only and is now gone.")
    print("Commit the files in", os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
