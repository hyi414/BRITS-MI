# BRITS-MI

Python software for longitudinal biomarker imputation and downstream outcome
analysis. The repository provides the clinical BRITS-MI interface, the
BRITS-MI-image fusion extension, simulation comparisons, and observed-cell
masking analysis for an authorized clinical dataset.

## Installation

Python 3.10 or later is required. From the repository root:

```bash
git clone https://github.com/hyi414/BRITS-MI.git
cd BRITS-MI
python -m pip install -e .
```

The manuscript clinical comparisons additionally require R and the `mice` and
`missForest` packages:

```r
install.packages(c("mice", "missForest"))
```

## Clinical Input

`UnifiedBRITSMI` accepts biomarker values and observation masks
(`N x T x J`), visit times (`N x T`), static covariates (`N x P`), and a binary
training outcome (`N`). Here, `N`, `T`, and `J` denote subjects, visits, and
biomarkers. The mask, not a value of zero, identifies missing observations.

```python
from brits_mi import UnifiedBRITSMI, UnifiedConfig

config = UnifiedConfig(seed=21, bootstrap_members=5, residual_mode="correlated")
imputer = UnifiedBRITSMI(config, task="association")
imputer.fit(values, mask, times, static, outcome)
imputed_datasets = imputer.sample(20, seed=22)
```

The recurrent model learns from observed longitudinal data and the specified
downstream outcome. Sampling retains observed values and returns multiple
imputed trajectories. Optional `context` supplies additional subject-level
features. Association imputations condition on an observed outcome.
Use `task="prediction"` for label-free new-patient imputation and prediction.

See the [unified example](examples/unified_clinical.py),
[input specification](docs/data_contract.md), and
[methodology](docs/methodology.md). The unified recurrent implementation does
not use a missForest mean anchor or a true-coefficient correction. Historical
implementations and their saved results remain under `simulation/clinical/`
and `reproducibility/`; they must not be attributed to the new engine.

## Clinical + Image Input

The same recurrent engine accepts subject-level image features through
`context`. The current benchmark extracts 17 descriptors from synthetic
grayscale ultrasound-like images. Labels are coded `0`, `1`, and `2`;
`-1` marks an unavailable training label.

```python
from brits_mi import UnifiedBRITSMI, UnifiedConfig
from brits_mi.unified import fusion_features

fusion = UnifiedBRITSMI(
    UnifiedConfig(seed=21, bootstrap_members=5, residual_mode="correlated"),
    task="prediction", feature_function=fusion_features)
fusion.fit(train_values, train_mask, train_times, train_covariates,
           train_labels, context=train_image_features, lengths=train_lengths)
probabilities = fusion.predict_proba(
    values=test_values, mask=test_mask, times=test_times, static=test_covariates,
    context=test_image_features, lengths=test_lengths, m=20)
```

Image information enters both recurrent paths, current-cell predictions,
and the fusion head. No test outcomes are required. The new benchmark uses
a common refitted neural head and allows unlabeled test covariates in
imputation to accommodate default missForest. Image MICE and missForest use
the same available clinical covariates and image descriptors, without an
outcome column. BRITS-MI uses training labels only as supervised training
targets. All methods receive 20 final classifier fits: single-imputation
methods reuse their imputed data with the same 20 classifier seeds. See the
[recurrent methodology](docs/methodology.md#recurrent-clinical-image-extension).
The earlier tree-based `BRITSMIImage` remains available for archived results;
it is not this recurrent implementation.

## Simulation Comparisons

The main experiment launcher compares BRITS-MI with MICE, missForest, mean
imputation, complete-case analysis, and full-data analysis using matched
synthetic datasets:

```bash
python simulation/unified/run_benchmark.py --kind clinical \
  --n 500 --runs 200 --m 20 --missing-percentages 20 40 60 \
  --bootstrap-members 5 --residual-mode correlated \
  --output results_local/clinical
```

The command starts a run. Use `--kind image --seed-start 930000` for the recurrent fusion
comparison. Outputs record source/configuration hashes, matched input hashes,
valid and invalid fits, and completed counts. The
[unified protocol](simulation/unified/README.md) separates development from
200-seed evaluation. Archived launchers remain under
[`simulation/`](simulation/README.md).

## Main Outputs

| Function or runner | Output |
| --- | --- |
| `UnifiedBRITSMI.fit(...)` | Recurrent ensemble, training history, and validation residual covariance |
| `UnifiedBRITSMI.sample(M, ...)` | List of `M` imputed arrays, each shaped `N x T x J` |
| `pool_logistic_regressions(...)` | Pooled coefficients, standard errors, confidence intervals, and imputation counts |
| `UnifiedBRITSMI.predict_proba(...)` | Mean class probability vectors across imputations |
| Clinical simulation runner | Per-run coefficient estimates, bias, MSE, coverage, imputation RMSE, trajectory-summary RMSE, and aggregate summaries |
| Image simulation runner | AUROC, AUPRC, classification loss, imputation error, and run configuration |
| Real-data masking runner | Source-relative coefficient estimates and reference-interval inclusion, masked-cell and trajectory errors, apparent classification metrics, and training/comparator logs |

Regression estimates use Rubin's pooling rules. For prediction from multiple
imputed datasets, probability vectors are averaged before calculating
classification metrics; AUROC and AUPRC are not pooled as regression coefficients.

## Real-Data Masking Analysis

[`validation/`](validation/README.md) contains the analysis code and required
input schema. Supply an authorized, analysis-ready cohort **outside this
repository**. No dataset, patient identifiers, imputed patient records, or
cohort-trained weights are distributed.

```bash
python simulation/unified/run_benchmark.py --kind real \
  --input-dir /secure/analysis_ready_cohort \
  --expected-cohort-size 3995 \
  --output /secure/brits_mi_validation \
  --runs 200 --m 20 --missing-percentages 20 40 60 \
  --bootstrap-members 5 --residual-mode correlated
```

Every imputer is fitted after masking. Its coefficient reference is the
pre-mask model, not a known population parameter. Apparent classification is
reported separately. The current TE-only source contains 3,995 participants
after excluding two MRE-coded records. This existing validation runner uses
the 2,209-participant subset meeting longitudinal-reference requirements.
It does not implement or establish results for the newer full-cohort,
18-coefficient analysis specification. Source and analysis denominators must
not be interchanged. Private validation data and outputs are not released.
See `validation/unified_validation.py` and the
[methodology](docs/methodology.md#comparators-and-real-data-masking).

## Versioned Evidence

The recurrent clinical evaluation completed 200 independent runs per setting.
Its results and individual-coefficient diagnostics are available in the
[aggregate results bundle](results/unified_recurrent_20261027/README.md).
They do not establish uniform superiority over competing methods.

Version `0.4.0` corrects the image comparison's outcome-information policy and
equalizes the number of classifier fits. The previous 200-run image output is
retained as an archived comparison, not relabeled as the corrected result.
The conditional-mean recurrent engine is unchanged. The newer joint-likelihood
posterior-sampling pilot remains a separate development method, not this release.
See the [release specification](docs/release_040.md).

## Development

```bash
python -m pip install -e '.[dev]'
pytest
```

See [contributing](CONTRIBUTING.md), [reproducibility](docs/reproducibility.md),
[data policy](DATA_POLICY.md), and [citation metadata](CITATION.cff).
Released under the [MIT license](LICENSE). Research use only.
