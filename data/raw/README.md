# `data/raw/` — immutable input files

This directory holds the raw inputs to the VEPS pipeline. **Do not write to
it after initial ingestion.** Everything derived (features, training sets,
predictions, models) lives elsewhere under `data/`.

## Files

| File                        | Required | Notes                                       |
| --------------------------- | -------- | ------------------------------------------- |
| `cve_observations.csv`      | yes      | Per-CVE, per-day exploitation observations  |
| `cve_mentions.json`         | no       | Per-CVE, per-day mention counts             |
| `nvd/`                      | yes      | NVD JSON feeds (`veps download` populates)  |

If `cve_mentions.json` is absent, the pipeline logs at INFO and proceeds
with no mention features. If it is present, it is validated against the
schema in [`src/veps/data/schemas.py`](../../src/veps/data/schemas.py) at
load time and the pipeline raises on any mismatch.

## Sample files

Synthetic 2–3 row examples are committed for reference and to demonstrate
the expected shapes:

- [`cve_observations.sample.csv`](./cve_observations.sample.csv)
- [`cve_mentions.sample.json`](./cve_mentions.sample.json)

The schemas they conform to are in
[`src/veps/data/schemas.py`](../../src/veps/data/schemas.py).
