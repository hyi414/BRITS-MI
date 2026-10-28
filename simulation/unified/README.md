# Unified recurrent benchmark

This is the revised comparison, not a relabeling of archived results. One
recurrent BRITS-GRU implementation supplies clinical and image-conditioned
imputations. R package MICE and missForest receive the same available
predictors, including observed outcomes for association analyses and the
same image descriptors for fusion analyses.

## Fixed evaluation settings

- 200 evaluation seeds per scenario and missing percentage; 500 subjects;
  20 stochastic imputations for BRITS-MI and MICE.
- Clinical scenarios: `mild_harder`, `baseline_harder`, `strong_harder`.
  Missing percentages: 20%, 40%, 60%, targeted among the final three visits.
  Actual percentages over late and all visits need not be identical.
- Larger-sample sensitivity: baseline scenario, 2,000 subjects, 200 seeds.
- MAR sensitivity: baseline scenario, 500 subjects, 200 seeds; only the
  observed outcome, static covariates, and fully observed early history
  determine late-cell masking. This is separate from the original mechanism.
- Image extension: archived alternative-ultrasound DGP, 500 subjects,
  200 seeds at each missing percentage, recurrent image-conditioned imputation,
  and one common neural fusion-head architecture.
- Real application: the user authorized the older source with 3,995 TE
  participants. The longitudinal reference analysis uses 2,209 eligible
  participants and 200 masks per percentage. These results remain separate
  from the manuscript's 3,977-participant cohort.

### Irregular trajectories and masking

The clinical DGP is imported from the archived
`generate_locked_dataset` function, not replaced by the simplified public
example. It has six unequal visit times, fixed within each scenario:

| Scenario | Visit times | Late acceleration | Pulse | Noise | Missingness abnormality | Missingness slope |
|---|---|---:|---:|---:|---:|---:|
| Mild | 0, .10, .27, .50, .77, 1 | 1.05 | 1.05 | 1.05 | 1.10 | 1.15 |
| Baseline | 0, .07, .22, .46, .73, 1 | 1.30 | 1.25 | 1.15 | 1.40 | 1.50 |
| Strong | 0, .03, .14, .37, .67, 1 | 1.65 | 1.45 | 1.25 | 1.80 | 2.00 |

The table entries after visit times are multipliers. Times are not random
gamma-gap draws in this locked DGP. Subject-specific severity, slopes, late
activity, acceleration, pulse locations and widths, and measurement errors
generate heterogeneous trajectories. The logistic outcome uses the 13
terms listed in `docs/methodology.md`. The original cell-masking probability
depends on outcome, diabetes, hypertension, visit, marker, current biomarker
abnormality, terminal slopes, and latent late activity. Its intercept targets
the expected late-cell missing percentage, but masking is sampled at all
six visits. It is nonmonotone and depends on unavailable values.

### Development and uncertainty selection

Development seeds 821000--821009 tested IID residual noise, correlated
residual noise, and correlated noise with five subject-bootstrap fits at
20%, 40%, and 60% missingness. These seeds are excluded from evaluation.
The bootstrap/correlated scheme is carried forward because it improved
high-missingness interval calibration and coefficient MSE within BRITS-MI;
it did not dominate MICE and missForest on every metric. Use this same
uncertainty scheme for every evaluation scenario. Do not select it again
using evaluation seeds. Numerical fixes to padded-time scaling and residual
covariance symmetry were checked separately before freezing evaluation code.

## Run

```bash
python simulation/unified/run_benchmark.py --kind clinical \
  --output results_local/clinical --n 500 --runs 200 --m 20 \
  --scenarios mild_harder baseline_harder strong_harder \
  --missing-percentages 20 40 60 --seed-start 910000 \
  --bootstrap-members 5 --residual-mode correlated --workers 2
```

Use `--kind image` for the image benchmark. Real data requires `--kind real`,
`--input-dir`, and the confirmed input schema. The default cohort-size guard
is 3,977; pass `--expected-cohort-size 3995` for the authorized older source.
This check does not establish participant eligibility by itself. Real
source-history eligibility and cohort counts are reported separately.

## Outputs and interpretation

| File | Contents |
|---|---|
| `manifest.json` | Code hash, package versions, design, seeds, input hash if applicable |
| `status.json` | Completed/failed jobs and current process |
| `jobs/*/result.json` | All methods, data/mask hashes, metric values, training diagnostics, R warnings |
| `jobs/*/error.json` | Uncaught job error; never interpreted as a completed valid fit |
| `metrics_by_run.csv` | Per-run outcomes and explicit invalid fits |
| `validity_summary.csv` | Valid, invalid, and attempted denominators |
| `coefficient_by_run.csv` | Individual coefficient estimates, SEs, variances, and interval indicators |
| `term_summary.csv` | Term-specific bias, MSE, coverage, empirical/model SE, MCSE |
| `overall_association.csv` | Averages across the prespecified coefficient terms |
| `summary.csv` | Metric means, SDs, MCSEs, and valid/attempted counts |
| `paired_comparisons.csv` | Matched-seed differences, intervals, paired tests, Holm-adjusted p-values |

Systematic bias is the mean signed error across independent runs; overall
absolute bias averages the absolute term-specific biases, not within-run
absolute errors. Overall coverage averages each replicate's proportion of
covered terms, so its MCSE uses variability across replicates, not a
binomial denominator treating correlated terms as independent.

All methods use the same reference SDs for standardized effects. MICE uses
default conditional models, five iterations, and 20 draws. Default
missForest supplies one imputation; model SEs do not include imputation
uncertainty for this baseline. Complete-case and full-data rows remain
distinct. Separation and non-identifiable fits are reported with denominators.

The fusion benchmark holds out labels but permits observed unlabeled test
covariates in every imputer. It is transductive. Average class probabilities
across imputations before calculating macro AUROC, macro average precision,
and log loss. Do not Rubin-pool AUC values.

Real-data intervals are evaluated against a pre-mask reference estimate,
not a known population coefficient. Natural missing values are not used as
known reconstruction targets. Keep all participant-level input and output
outside this repository.
