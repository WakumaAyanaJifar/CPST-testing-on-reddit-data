# Testing the Cyber-Physical-Social-Thinking quadspace on Reddit data

Code, lexicons, de-identified author-level features and results for:

> "Are the cyber-physical-social-thinking spaces separately measurable? A
> validation protocol and evidence from Reddit behavioural traces," manuscript
> submitted, 2026.

The study asks whether the four CPST spaces (cyber, physical, social and
thinking) can be measured separately from platform behaviour. It builds
author-level features for 2,595 Reddit authors across eight communities
(2023–2025) and tests reliability, separability, coupling, known-groups
validity, and the circadian inference on which the physical space depends.
On these data, method variance dominates the CPST spaces: the CPST allocation is
indistinguishable from a partition by data source, and a multitrait-multimethod
test with text as a second method fails for all three spaces it covers. The
physical block contains two nearly unrelated components, daily timing and
posting tempo, and individual circadian offsets are validated against simpler
offset assignments.

## Repository contents

| Folder | Contents |
|---|---|
| `scripts/` | The full pipeline, from collection to figures (see below). |
| `lexicons/` | Every lexicon and keyword list used, extracted from the code. |
| `data/` | De-identified author-level features, and the data dictionary. |
| `results/` | Machine-readable results (JSON, CSV) and LaTeX tables from the paper's runs. |
| `figures/` | Every figure in the paper, as PDF and PNG. |

### Pipeline

| Script | Role | Needs the private database? |
|---|---|---|
| `fomo_collector_v6.py` | Four-stage collection through the Reddit Data API | Creates it |
| `recover_and_expand.py` | Thread completion, author expansion, control cohort | Yes |
| `extract_cpst_features.py` | Builds the 71 features; `--half 1/2` for split-half files; `--rematch` for matching | Yes |
| `prepare_public_data.py` | Writes the de-identified files in `data/` | No (reads the private feature files) |
| `analyze_spaces_v2.py` | Separability, canonical coupling, offset distribution, record-level validation | Only for the record-level test |
| `analyze_validity.py` | Reliability, partition test, known groups, redundancy | No |
| `mh_robustness.py` | Cluster bootstrap, anchor sweep, placebo sweep | Yes |
| `make_figures.py` | Figs. 1–3 and Table I | Figs. 2–3 and Table I only |
| `revision_analyses.py` | Physical sub-groups, mapping sensitivity, half-sampling intervals, role-matched known groups, circadian subsample profile, automated-account screen | Only the subsample profile and account screen (`--skip-db` runs the rest) |
| `revision_b1_check.py` | Physical sub-groups with definitionally linked feature pairs excluded | No |
| `revision_b7_loadings.py` | Half-sample stability of canonical structure coefficients | No |
| `revision2_channel.py` | CPST vs data-channel partition, channel-restricted null, cross-channel features, shared-component exclusion, conservative redundancy, k = 4 benchmark, lexicon count | Only the lexicon count (`--skip-db` runs the rest) |
| `revision3_checks.py` | Paired CPST-minus-channel interval, multitrait-multimethod matrix with text as second method, rhythm/tempo construction checks, week-block reliability, circadian offset baselines | Yes (reads record text and timestamps) |
| `r4_power.py` | Power of rule R4 by simulation | No (synthetic data only) |
| `revision4_final.py` | Paired bootstrap of offset comparisons, social construction null, held-out replication by community block, MTMM positive control | Yes (reads record text and thread structure) |

## Reproducing the results from the public data

Requires Python 3.10 or later.

```bash
pip install -r requirements.txt
export CPST_BASE_DIR="$(pwd)/data"          # Windows PowerShell: $env:CPST_BASE_DIR = "$PWD\data"

python scripts/analyze_validity.py
python scripts/analyze_spaces_v2.py --skip-records
python scripts/analyze_spaces_v2.py --drop-counts --tag _nocounts --skip-records
python scripts/analyze_spaces_v2.py --drop-counts --drop-partial --tag _core --skip-records

# robustness analyses added in revision
python scripts/revision_analyses.py --skip-db
python scripts/revision_b1_check.py
python scripts/revision_b7_loadings.py
python scripts/revision2_channel.py --skip-db
python scripts/r4_power.py
```

Week-block split halves (reliability sensitivity check) are built from the
private database with:

```
python scripts/extract_cpst_features.py --half 1 --split week
python scripts/extract_cpst_features.py --half 2 --split week
```

Outputs are written to `data/paper1/`. They reproduce reliability,
separability, the physical sub-group analysis, canonical coupling and its
stability, known-groups validity (including the role-matched contrast),
redundancy, the partition test, the mapping-sensitivity checks and the offset
distribution. `revision_analyses.py` prints a self-check: its "as published"
row must read margin +0.0314 and corrected gaps C +0.057, P +0.019, S +0.180,
T +0.040.
Permutation and cross-validation steps are seeded and the public files keep
the private row order, so results match those in `results/`. Small
differences can arise only from different library versions.

The record-level circadian validation, the profile of its 1,290-author
subsample and the automated-account screen read individual records and cannot be
rerun from the public files. Their complete outputs are in
`results/mh_robustness.json`, `results/mh_placebo.json`,
`results/results_v2.json`, `results/revision_analyses.json`,
`results/revision2_channel.json`, `results/revision3_checks.json` and
`results/revision4_final.json`.

## What is not released, and why

Following the commitments in the paper (Section IV-E):

- **Record text.** A quoted sentence can be searched back to its author.
- **Platform record identifiers.** Each resolves through the Reddit API to the
  author's username and text. They are available to researchers under a
  data-use agreement; contact the corresponding author.
- **Author hashes and the hashing salt.** Public files use fresh random
  identifiers that cannot be linked to the private data.
- **Exact timestamps.** Each author's first and last posting time is removed,
  since together with record counts it could be matched against public
  archives.

To collect a comparable corpus yourself, set `REDDIT_CLIENT_ID` and
`REDDIT_CLIENT_SECRET` as environment variables and run the collection scripts.
Credentials are never stored in the code.


## Licence

Code is released under the MIT Licence (`LICENSE`). Data, lexicons, results and
figures are released under Creative Commons Attribution 4.0 (`LICENSE-DATA`).
Use of the data must also comply with Reddit's terms for the content from which
the features were derived.

