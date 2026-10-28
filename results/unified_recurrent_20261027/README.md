# Unified recurrent evaluation

This bundle contains **synthetic aggregate results only**. No patient data,
patient-level imputations, private cohort linkage, or trained patient models
are included. Recurrent evaluation code is preserved at commit `ae52dfa`.

| Folder | Subjects | Independent runs per scenario / missing percentage | Status |
| --- | ---: | ---: | --- |
| `clinical_n500_200` | 500 | 200 | Completed; 3 scenarios, 3 missing percentages |
| `clinical_n2000_200` | 2,000 | 200 | Completed; baseline, 3 missing percentages |
| `clinical_mar_200` | 500 | 200 | Completed; observed-history masking, 3 missing percentages |
| `image_200` | 500 | 200 | Archived comparison; see differences below |

Each table reports point estimates and Monte Carlo SE, with valid and attempted
denominators. `individual_coefficients.csv` retains all 13 clinical terms,
including empirical/model SE and within/between-imputation variance.
`paired_contrasts.csv` uses matched independent cohorts. Its p-values are
unadjusted, two-sided paired t-tests and must not be described as multiplicity
adjusted or as evidence of superiority where the difference favors a comparator.

Overall coverage first averages the 13 interval indicators within a cohort;
its Monte Carlo SE is calculated across cohorts. Overall mean absolute bias
averages absolute term-specific biases, not absolute errors from single runs.
Its Monte Carlo SE is a cohort bootstrap standard error (1,000 resamples).
Mean MSE, coverage and reconstruction errors use their across-cohort SEs.
Figures show estimates with 1.96 times Monte Carlo SE. They are not individual
coefficient intervals. All methods appear in tables, even when not plotted.

The archived image benchmark included observed training labels in MICE and
missForest and used unequal classifier-fit counts for single and multiple
imputation. It is not the corrected version 0.4.0 comparison. The corrected
run removes the comparator outcome column and uses 20 classifier fits for
every method, without changing the recurrent algorithm, DGP or seeds.

The newer joint-likelihood/HMC development pilot is also not this benchmark.
Clinical findings are mixed: lower coefficient MSE at n = 500 does not imply
lower bias or reconstruction error, and MICE performs better on coefficient
MSE in the n = 2,000 experiment. All findings are retained.

Open `report.html` locally to inspect tables and figures. `updated_figures.pdf`
contains the figure set with vector text. Result-file hashes and the completed
audit are recorded in `verification.json` and each experiment directory.

Rebuild from the original, private-workspace synthetic job outputs:

```bash
PYTHONPATH=src python simulation/unified/release_results.py \
  --evaluation /path/to/evaluation \
  --frozen-source /path/to/frozen_source \
  --output /path/to/rebuilt_aggregate_bundle
```
