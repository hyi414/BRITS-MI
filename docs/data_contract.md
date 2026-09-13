# Input data contract

BRITS-MI accepts one row of longitudinal history per subject after visits have
been ordered within subject. The public API uses five arrays.

| Object | Shape | Meaning |
|---|---:|---|
| `values` | subjects x visits x biomarkers | Numeric biomarker values; naturally missing cells may be `NaN`. |
| `mask` | subjects x visits x biomarkers | `True` only when the source value was measured. |
| `times` | visits, or subjects x visits | Nondecreasing elapsed times. Patient-specific times are supported. |
| `static` | subjects x covariates | Deployment-available clinical and demographic covariates. |
| `context` | subjects x features | Optional image summaries or learned image embeddings. |

`outcome` is supplied only to `fit` because it contributes to the training
loss. It is not accepted by `sample`, the label-free deployment method.

## Two masks must remain distinct

- The source mask `M` records genuine observation. Cells with `M=0` have no
  known target and never enter reconstruction loss.
- The artificial-holdout mask `R` is sampled only among cells with `M=1`.
  These temporarily hidden values provide training, validation, and evaluation
  targets. The code constructs `R` internally during fitting.

Measured cells are copied back after every completion, so BRITS-MI cannot
overwrite a source observation. Split subjects before estimating scaling
constants, residual scales, or model-selection criteria.

## Real-data boundary

This repository contains no patient-level records or identifiers. Convert a
governed cohort to the tensor contract inside the approved environment. Do not
commit clinical extracts, fitted weights derived from restricted data, or logs
that contain identifiers.
