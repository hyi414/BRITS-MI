# Reproducibility map

## Unified recurrent release (0.4.0)

| Component | Entry point |
| --- | --- |
| Shared BRITS-GRU engine and stochastic imputation | `src/brits_mi/unified.py` |
| Clinical coefficient fitting and reference scaling | `src/brits_mi/benchmark.py` |
| Recurrent image input and common fusion classifier | `src/brits_mi/recurrent_image.py` |
| Native R MICE/missForest interfaces | `src/brits_mi/comparators.py` and `src/brits_mi/r/impute.R` |
| Fixed synthetic experiment launcher | `simulation/unified/run_benchmark.py` |
| Verified aggregate results, tables and figures | `simulation/unified/release_results.py` |
| Private reference-eligible TE masking analysis | `validation/unified_validation.py` |

The completed clinical evaluation uses 200 cohorts per setting and M = 20.
The image comparison correction and the distinction from the separate
posterior-sampling development pilot are documented in
[release_040.md](release_040.md). See the
[current aggregate bundle](../results/unified_recurrent_20261027/README.md).

## Archived analysis map

The entries and counts below describe historical analyses, not the current
unified recurrent evaluation. They remain for reproducibility.

| Manuscript component | Repository entry point |
|---|---|
| RITS-GRU and bidirectional reconciliation | `src/brits_mi/model.py` |
| Artificial masking, weighted loss, validation, and stochastic draws | `src/brits_mi/imputer.py` |
| Optional estimating-score calibration | `src/brits_mi/calibration.py` |
| Irregular clinical-association DGP | `src/brits_mi/simulation.py` |
| Rubin pooling and recovery metrics | `src/brits_mi/analysis.py` |
| Ultrasound-like renderer and 17 image summaries | `src/brits_mi/ultrasound.py` |
| Simulation grid and machine-readable manifest | `src/brits_mi/runner.py` |
| Package-default R comparators | `simulation/R/default_comparators.R` |
| Manuscript clinical comparison launcher | `simulation/run_simulation.py` |
| Clinical benchmark implementation and dependencies | `simulation/clinical/scripts/` |
| Historical image comparison runner | `simulation/image/scripts/run_fast_image_fusion_nsim500.py` |
| Real-data masking code, without the cohort | `validation/run_validation.py` and `validation/README.md` |

The simulation scripts were consolidated under `simulation/`. Their source
hashes remain recorded in `reproducibility/source_manifest.json`, with updated
destinations. The date-stamped release inventory records the original release
paths and has not been rewritten. The former
`reproducibility/run_reported_clinical.py` command remains a compatibility entry
point. Reusable library code stays under `src/`.

## Archived experiment descriptions

- Primary association benchmark: 500 subjects, 200 independent runs per 20%,
  40%, and 60% target late-cell missingness rate, and 50 completed datasets for
  BRITS-MI and MICE. The reported benchmark used a shared fixed six-point
  non-equidistant grid (0.00, 0.07, 0.22, 0.46, 0.73, and 1.00), not
  patient-specific gamma-distributed visit times.
- Larger-sample sensitivity: 2,000 subjects and 100 independent runs per rate.
- Clinical-image benchmark: 500 subjects and 500 matched independent runs.
- Observed-cell masking analysis: 3,997 elastography examinations and 100
  repeated masks per rate; naturally missing A1c is not scored.

The public package provides an executable reference implementation and a
compact smoke configuration. Full paper-scale runs require substantial
compute. Raw clinical data are excluded; aggregate result tables and exact
release identifiers should be archived with the final publication.

The compact package API is not represented as a byte-for-byte reproduction of
every exploratory script used while the analysis was developed. See
`reproducibility/reported_pipeline/README.md` for the exact analysis map and
the role of the outcome-blind missForest anchor in the locked association
benchmark.
