# Step-by-step code walkthrough

## 1. Represent a longitudinal cohort

Prepare `values` with shape `(subjects, visits, biomarkers)`, a Boolean
`mask` of the same shape, nondecreasing visit `times`, subject-level `static`
covariates, and a subject-level `outcome`. A value is considered measured only
when its mask is true. Genuine missing values must never be inserted into an
artificial reconstruction target.

## 2. Fit the recurrent completion distribution

```python
from brits_mi import BRITSMultipleImputer, TrainingConfig

imputer = BRITSMultipleImputer(
    TrainingConfig(epochs=80, hidden_size=48, seed=20260904)
)
imputer.fit(values, mask, times, static, outcome, context=image_features)
```

During each epoch, the trainer hides a random subset of measured cells,
computes forward and backward RITS-GRU predictions, locks the remaining
measured cells, and optimizes reconstruction, directional consistency,
trajectory-summary, and downstream losses. Model selection uses separately
hidden validation cells.

## 3. Draw completed trajectories

```python
completed = imputer.sample(
    n_imputations=20,
    values=values,
    mask=mask,
    times=times,
    static=static,
    context=image_features,
    seed=20260905,
)
```

Each draw perturbs only missing cells around the frozen conditional mean using
validation-holdout residual scales. Measured cells are copied back unchanged.

## 4. Refit and pool an association model

```python
from brits_mi.analysis import pool_logistic_regressions

pooled = pool_logistic_regressions(completed, times, static, outcome)
print(pooled.loc[["late_ast", "slope_alt", "slope_platelet"]])
```

The downstream model is refitted once per completed dataset. Point estimates
and within- and between-imputation variance are combined with Rubin's rules.

## 5. Optional explanatory association calibration

```python
from brits_mi import AssociationCalibrationConfig

calibrated, diagnostics = imputer.sample_association_calibrated(
    n_imputations=20,
    outcome=outcome,
    coefficients=reference_coefficients,
    intercept=reference_intercept,
    calibration=AssociationCalibrationConfig(step_size=0.08, n_steps=1),
)
```

This operation uses outcomes and a prespecified coefficient target. It is for
an explanatory association analysis and is not the label-free interface for a
new patient. Backtracking accepts a bounded missing-cell update only when the
standardized score discrepancy does not increase.
