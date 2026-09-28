# Real-Data Masking Analysis

Code for the manuscript's repeated observed-cell masking analysis. The cohort
is not distributed. Prepare inputs and run the analysis inside an authorized
environment; both input and output directories must be outside this checkout.

## Inputs

Provide two files in `--input-dir` with the same examination order. Use
consecutive `row_index` values from 0 through N-1, not patient identifiers.
The loader checks array shapes, channel order, row indices, and coding. The
local cohort-preparation process remains responsible for correct linkage.

**`longitudinal.npz`**, saved as numeric/Unicode arrays, without pickle objects:

| Key | Shape | Definition |
| --- | --- | --- |
| `values` | N x 6 x 6 | Biomarker measurements on the analysis scale, before artificial masking |
| `mask` | N x 6 x 6 | 1 for a source-observed value; 0 for natural missingness |
| `static` | N x P | Finite, analysis-ready clinical/demographic covariates; no outcome or identifiers |
| `row_index` | N | Consecutive row indices shared with the clinical file |
| `marker_names` | 6 | Exactly `a1c`, `sbp`, `dbp`, `ast`, `alt`, `platelet`, in that order |

The six bins are ordered from oldest to most recent before examination. The
model uses bin position and observed history to represent timing. This analysis
does not accept arbitrary timestamps as though they were these fixed bins.
Zeros in `values` can be measured values; only `mask` determines availability.

**`clinical.csv`** must contain exactly these columns:

| Columns | Coding |
| --- | --- |
| `row_index` | 0, ..., N-1, in array order |
| `outcome_3class` | `F0-F1`, `F2`, or `F3-F4` |
| `FIB4_log`, `FIB4_Age`, `bmi_consolidated` | Numeric log FIB-4, age, and body mass index |
| `Diabetes_RF`, `HTN_RF`, `HLD_RF` | Yes/No or 1/0; no missing values |
| `Sex` | Male/Female for the specified model |
| `Study_Type` | Examination modality; strings containing MRE identify magnetic resonance elastography |

Supply the same clinical transformations and static-feature design used in the
analysis being reproduced. The input adapter replaces institution-specific
workbook linkage, not the scientific data-preparation decisions. Missing numeric
clinical predictors use the source-cohort median, as in the original model.

## Run

```bash
python validation/run_validation.py \
  --input-dir /secure/analysis_ready_cohort \
  --output /secure/brits_mi_validation \
  --repeats 500 --missing-percentages 20 40 60 \
  --methods brits mice missforest mean
```

For a software check, use `--repeats 1 --epochs 1 --ensemble-size 2 --methods
brits mean`. Such a run is not an assessment of statistical performance.
MICE and missForest require R and the corresponding packages. The command
rejects nonempty output directories rather than overwriting previous analyses.

## Analysis Procedure

1. Fit a common pre-mask binary outcome model for F3-F4 versus the other stages.
   It contains 19 non-intercept terms: log FIB-4, age, body mass index, diabetes,
   hypertension, hyperlipidemia, sex, modality, log FIB-4 by diabetes and
   hypertension by sex interactions, and AST/ALT/platelet late levels, terminal
   changes, and source missing fractions. Each fit standardizes its design.
2. Train a bootstrap ensemble of ten history-aware bidirectional GRU models on
   the source cohort. Artificial training masks target measured AST, ALT, and
   platelet cells. Reconstruction, directional consistency, observed-cell,
   trajectory-summary, association-score, and binary outcome losses update the
   imputer. Default settings are 12 epochs, 32 hidden units, and batch size 256.
   Summary, association-score, and outcome weights are 0.6666666667, 0.5, and 0.05.
3. Independently mask measured AST, ALT, and platelet cells at each requested
   percentage. All methods receive the same repeat-specific mask. Naturally
   missing cells remain separate and have no known evaluation target.
4. BRITS-MI returns one imputed dataset from each bootstrap model, with no added
   residual noise in this configuration. MICE returns ten imputations using R
   defaults with five iterations. R missForest returns one imputed dataset.
   Mean imputation estimates each visit-marker mean from the unmasked cells.
   Comparator predictor frames contain biomarkers, masks, and static covariates,
   but not the outcome.
5. Refit the specified association model and pool coefficient estimates with
   within- and between-imputation variance. Calculate deviation from the pre-mask
   reference and whether each nominal 95% interval contains that reference.
   Refit multinomial classifiers, average probabilities across imputations, and
   calculate apparent AUROC, AUPRC, and log loss.

The GRU ensemble is trained before evaluation masks are drawn, whereas the
comparators are fitted to masked data. This is the archived same-cohort design,
not independent held-out validation. The code does not estimate population
coverage or performance on new patients from repeated masks of one cohort.

## Outputs

| File | Contents |
| --- | --- |
| `config.json` | Configuration and hashes of the supplied inputs; no private file paths |
| `reference_coefficients.csv` | Pre-mask model coefficients |
| `training_history.csv` | Ensemble member, epoch, and loss components |
| `metrics_by_repeat.csv` | Association, reconstruction, and apparent classification metrics |
| `coefficients_by_repeat.csv` | All 19 terms, reference estimates, pooled estimates/SE, deviations, and interval inclusion |
| `summary.csv` | Mean, SD, Monte Carlo SE, and valid/attempted repeat counts by metric |
| `term_summary.csv` | Per-term deviation, MSE, coverage, model SE, and repeated-mask SD |
| `package_audit.csv` | R package versions and comparator diagnostics |

For compatibility with the archived analysis, overall association metrics use
the 12 prespecified terms in `evaluation.DISPLAY_TERMS`; all 19 are retained in
the coefficient file. `effect_abs_bias` is the mean absolute deviation within
a masking repeat, not absolute systematic bias across independently sampled
cohorts. `term_summary.csv` also reports mean signed deviations across masks.

Masked-cell NRMSE divides the RMSE by the SD of the pooled hidden measurements.
Trajectory-summary NRMSE compares source-observed means, late means, and
last-minus-first changes after replacing only artificially hidden cells. It
uses the pooled SD of those source summaries. These are not marker-specific
standardizations. Interval inclusion refers to the source model, not known truth.

## Code and Data Boundary

`brits.py` contains the recurrent engine and training objective; `evaluation.py`
contains the original masking, outcome-model, and pooling routines. Their source
symbols and extraction hashes are recorded in `source_manifest.json`.
`data.py` and `run_validation.py` provide the portable input and execution layer.

No dataset, patient-level predictions, masks, imputed records, or fitted weights
are written by this public runner. Outputs are aggregate or coefficient-level,
but still require institutional disclosure review before sharing. Temporary
MICE/missForest inputs are created under the system temporary directory and
removed after each call; set `TMPDIR` to approved encrypted storage when required.
See the [data policy](../DATA_POLICY.md).
