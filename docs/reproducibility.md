# Reproducibility map

| Manuscript component | Repository entry point |
|---|---|
| RITS-GRU and bidirectional reconciliation | `src/brits_mi/model.py` |
| Artificial masking, weighted loss, validation, and stochastic draws | `src/brits_mi/imputer.py` |
| Optional estimating-score calibration | `src/brits_mi/calibration.py` |
| Irregular clinical-association DGP | `src/brits_mi/simulation.py` |
| Rubin pooling and recovery metrics | `src/brits_mi/analysis.py` |
| Ultrasound-like renderer and 17 image summaries | `src/brits_mi/ultrasound.py` |
| Simulation grid and machine-readable manifest | `src/brits_mi/runner.py` |
| Package-default R comparators | `reproducibility/R/default_comparators.R` |

## Locked experiment descriptions

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
