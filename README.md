# Testing the Cyber-Physical-Social-Thinking quadspace on Reddit data

Code, lexicons, de-identified author-level features and results for:

> "Empirical measurement and separability testing of
the cyber-physical-social-thinking (CPST) spaces
using large-scale social platform behavioural data," manuscript under review, 2026.

The study asks whether the four CPST spaces (cyber, physical, social and
thinking) can be measured separately from platform behaviour. It builds
author-level features for 2,595 Reddit authors across eight communities
(2023–2025) and tests reliability, separability, coupling, known-groups
validity, and the circadian inference on which the physical space depends.

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

## Reproducing the results from the public data

Requires Python 3.10 or later.

```bash
pip install -r requirements.txt
export CPST_BASE_DIR="$(pwd)/data"          # Windows PowerShell: $env:CPST_BASE_DIR = "$PWD\data"

python scripts/analyze_validity.py
python scripts/analyze_spaces_v2.py --skip-records
python scripts/analyze_spaces_v2.py --drop-counts --tag _nocounts --skip-records
python scripts/analyze_spaces_v2.py --drop-counts --drop-partial --tag _core --skip-records
```

Outputs are written to `data/paper1/`. They reproduce reliability (Table II),
separability (Table III), canonical coupling (Table IV), known-groups validity
(Table V), redundancy, the partition test and the offset distribution.
Permutation and cross-validation steps are seeded and the public files keep
the private row order, so results match those in `results/`. Small
differences can arise only from different library versions.

The record-level circadian validation (Tables VI–VII, Figs. 9–12) reads the
text of individual records and cannot be rerun from the public files. Its
complete outputs are in `results/mh_robustness.json`,
`results/mh_placebo.json` and `results/results_v2.json`.

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

## Citation

If you use this code or data, please cite the paper above. Citation metadata is
in `CITATION.cff`.

## Licence

Code is released under the MIT Licence (`LICENSE`). Data, lexicons, results and
figures are released under Creative Commons Attribution 4.0 (`LICENSE-DATA`).
Use of the data must also comply with Reddit's terms for the content from which
the features were derived.

## Contact

Jifar Wakuma Ayana, School of Computer and Communication Engineering,
University of Science and Technology Beijing.
Email:d202561032@xs.ustb.edu.cn
