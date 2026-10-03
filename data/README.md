# Data

This folder holds the de-identified author-level features. Column definitions
are in `DATA_DICTIONARY.md`.

| File | Contents |
|---|---|
| `cpst_features.csv` | Features computed on each author's full 2023–2025 record set. |
| `cpst_features_h1.csv` | The same features computed on odd-numbered records only. |
| `cpst_features_h2.csv` | The same features computed on even-numbered records only. |

The files were produced by `scripts/prepare_public_data.py`, which replaces
private author hashes with random identifiers and removes exact timestamps.
Row order and matched-pair numbers are kept as in the private files, so the
seeded analyses reproduce the published numbers exactly.

Point the analysis scripts at this folder with the `CPST_BASE_DIR`
environment variable (see the main README).
