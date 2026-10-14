# ckks-tree-inference-exactness

Code, stored results and generated tables and figures for a manuscript in preparation:

> **Where Approximate Node Comparison Fails and What Exactness Costs in Non-Interactive CKKS Inference with Gradient-Boosted Trees**

The study measures how a polynomial approximation of the node comparison behaves when a gradient-boosted tree ensemble is evaluated on encrypted inputs with CKKS, and what an exact comparison costs on quantized inputs. It compares an interpolation circuit, whose client must know the thresholds, with a thermometer-bit protocol, in which the client sends only bits of its binned inputs. Results come from three backends, always stated: plaintext XGBoost, a NumPy simulation of the circuit's arithmetic, and real ciphertexts (Microsoft SEAL through TenSEAL).

The tasks are proxies built from MIMIC-IV-ED and MIMIC-IV. No result here is a clinical validation.

## Data: not included

The cohorts are built from MIMIC-IV-ED 2.2 and the MIMIC-IV hospital module 3.1, which are credentialed PhysioNet data that cannot be redistributed. This repository contains no patient-level data: no cohort or feature tables, no stay or patient identifiers, and no models trained on the data. Everything under `results/` is aggregate (counts, rates, summary statistics, byte sizes and timings); before publication the files were scanned by script for identifier fields and per-record arrays.

To re-run the pipelines you need your own PhysioNet access. `src/mimic_paths.py` looks for the MIMIC-IV-ED and MIMIC-IV files, and the cohort scripts also use the `mimic4ed-benchmark` scripts, in a sibling folder `module2/mimic4ed-benchmark-main` relative to the parent of this repository; that folder is not part of this repository.

## What is here

| Path | Contents |
|---|---|
| `src/` | Cohort and feature construction, plaintext training, the CKKS circuits and their simulator, the final-study scripts (`m1c_*`), evidence-freeze, table, figure and macro generators, and the verifier |
| `experiments/`, `configs/`, `reproduce.sh`, `run_thermo_final.sh` | Orchestration of the stages |
| `environment/` | Pinned requirements and `setup_venvs.sh` (Python 3.11 on WSL2 Ubuntu; TenSEAL 0.3.17; separate environments for the TFHE and lattice-estimator stages) |
| `results/` | Stored result files, one JSON per script run, and the evidence manifest `results/m1_evidence_manifest.json` |
| `paper_assets/` | Generated LaTeX tables, number macros, figures, and a provenance record for every recorded value |
| `paper_draft/floats/` | Figure captions that the verifier reads |
| `data/` | Two small public inputs only: the polynomial coefficients (`poly_coeffs_v1.json`) and the NDC to ATC crosswalk obtained from the public RxNav service (`atc_crosswalk.json`) |

## Check that every reported number comes from stored results

```
python src/verify_paper_assets.py
```

This replays every value recorded in `paper_assets/` against the stored files in `results/` and fails if a number, a label or a caption does not match. It needs no MIMIC data. Run from the repository root.

## Notes

- `reproduce.sh` and `experiments/run_reproducible_m1.py` were written for a layout in which the derived cohort tables sat in `data/`. Those tables are patient-level and are not published, so the default start point does not exist here; use `--from-scratch`, which rebuilds them from the raw files.
- The evidence manifest also lists the hashes of the data files and models that are not published; those entries cannot be checked without the credentialed data.
- Timings were taken on one desktop machine without controlled host load; they show relative costs, not benchmarks of the libraries.

## Licence

MIT, see `LICENSE`. The licence covers the code in this repository. MIMIC-IV data are governed by the PhysioNet credentialed data use agreement.
