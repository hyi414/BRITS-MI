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

`BRITSMultipleImputer` accepts biomarker values and observation masks
(`N x T x J`), visit times (`N x T`), static covariates (`N x P`), and a binary
training outcome (`N`). Here, `N`, `T`, and `J` denote subjects, visits, and
biomarkers. The mask, not a value of zero, identifies missing observations.

```python
from brits_mi import BRITSMultipleImputer, TrainingConfig

imputer = BRITSMultipleImputer(TrainingConfig(seed=21))
imputer.fit(values, mask, times, static, outcome)
imputed_datasets = imputer.sample(n_imputations=20, seed=22)
```

The recurrent model learns from observed longitudinal data and the specified
downstream outcome. Sampling retains observed values and returns multiple
imputed trajectories. Optional `context` supplies additional subject-level
features. New-patient sampling does not require an outcome label.

See the [clinical example](examples/run_clinical_association.py),
[input specification](docs/data_contract.md), and
[methodology](docs/methodology.md). The manuscript clinical runner also applies
the specified missForest conditional-mean anchor; it is provided separately
under [`simulation/clinical/`](simulation/clinical/).

## Clinical + Image Input

`BRITSMIImage` accepts longitudinal values, masks, times, sequence lengths,
clinical covariates, and grayscale images (`N x 1 x 24 x 24`). Training labels
are coded `0`, `1`, and `2` for the three outcome categories.

```python
from brits_mi.image_fusion import BRITSMIImage, ImageFusionConfig

fusion = BRITSMIImage(ImageFusionConfig(seed=21, prototype_alpha=0.0))
fusion.fit(train_values, train_mask, train_times, train_lengths,
           train_covariates, train_images, train_labels)
imputed_values = fusion.complete(
    test_values, test_mask, test_times, test_lengths, test_covariates, test_images)
probabilities = fusion.predict_proba(
    test_values, test_mask, test_times, test_lengths, test_covariates, test_images)
```

This extension uses image-conditioned tree regressors for imputation and an
ExtraTrees classifier for feature-level fusion. It produces a single imputed
dataset and class probabilities; it is not the recurrent multiple-imputation
engine. See [image inputs, feature extraction, and fusion](docs/image_fusion.md).

## Simulation Comparisons

The main experiment launcher compares BRITS-MI with MICE, missForest, mean
imputation, complete-case analysis, and full-data analysis using matched
synthetic datasets:

```bash
python simulation/run_simulation.py --experiment clinical \
  --subjects 500 --runs 500 --missing-percentages 20 40 60 \
  --methods brits mice missforest baselines \
  --output results_local/clinical --execute
```

Omit `--execute` to inspect the commands without starting a run. The
[`simulation/`](simulation/README.md) directory contains the clinical and image
experiment scripts, sensitivity analyses, R comparator interfaces, and output
definitions. Reusable model and data-generation functions remain in `src/`.

## Main Outputs

| Function or runner | Output |
| --- | --- |
| `BRITSMultipleImputer.fit(...)` | Fitted recurrent imputer, training history, and residual-scale calibration |
| `BRITSMultipleImputer.sample(...)` | List of `M` imputed arrays, each shaped `N x T x J` |
| `pool_logistic_regressions(...)` | Pooled coefficients, standard errors, confidence intervals, and imputation counts |
| `BRITSMIImage.complete(...)` | Imputed biomarker array, shaped `N x T x 3` |
| `BRITSMIImage.predict_proba(...)` | Three-class probabilities, shaped `N x 3` |
| Clinical simulation runner | Per-run coefficient estimates, bias, MSE, coverage, imputation RMSE, trajectory-summary RMSE, and aggregate summaries |
| Image simulation runner | AUROC, AUPRC, classification loss, imputation error, and run configuration |
| Real-data masking runner | Source-relative coefficient estimates and interval coverage, masked-cell and trajectory errors, apparent classification metrics, and training/comparator logs |

Regression estimates use Rubin's pooling rules. For prediction from multiple
imputed datasets, probability vectors are averaged before calculating
classification metrics; AUROC and AUPRC are not pooled as regression coefficients.

## Real-Data Masking Analysis

[`validation/`](validation/README.md) contains the analysis code and required
input schema. Supply an authorized, analysis-ready cohort **outside this
repository**. No dataset, patient identifiers, imputed patient records, or
cohort-trained weights are distributed.

```bash
python validation/run_validation.py \
  --input-dir /secure/analysis_ready_cohort \
  --output /secure/brits_mi_validation \
  --repeats 500 --missing-percentages 20 40 60
```

This reproduces the same-cohort observed-cell masking design. Its coefficient
reference is the pre-mask model, not a known population parameter. See the
[validation protocol](validation/README.md) for training, masking, and outputs.

## Development

```bash
python -m pip install -e '.[dev]'
pytest
```

See [contributing](CONTRIBUTING.md), [reproducibility](docs/reproducibility.md),
[data policy](DATA_POLICY.md), and [citation metadata](CITATION.cff).
Released under the [MIT license](LICENSE). Research use only.
