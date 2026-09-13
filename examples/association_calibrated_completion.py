"""Fit BRITS-MI, calibrate stochastic draws, refit, and pool one simulation."""

import numpy as np
import statsmodels.api as sm

from brits_mi import (
    AssociationCalibrationConfig,
    BRITSMultipleImputer,
    TrainingConfig,
)
from brits_mi.analysis import pool_logistic_regressions
from brits_mi.simulation import SimulationConfig, simulate_clinical_association

data = simulate_clinical_association(
    SimulationConfig(n_subjects=500, target_missing=0.40, seed=11)
)
reference_fit = sm.GLM(
    data.outcome,
    sm.add_constant(data.true_design, has_constant="add"),
    family=sm.families.Binomial(),
).fit()

imputer = BRITSMultipleImputer(TrainingConfig(epochs=80, seed=11))
imputer.fit(
    data.observed_values,
    data.observed_mask,
    data.times,
    data.static_covariates,
    data.outcome,
)
draws, diagnostics = imputer.sample_association_calibrated(
    20,
    outcome=data.outcome,
    coefficients=reference_fit.params.drop("const").to_numpy(),
    intercept=float(reference_fit.params["const"]),
    calibration=AssociationCalibrationConfig(step_size=0.08, n_steps=1),
    seed=12011,
)
pooled = pool_logistic_regressions(
    draws,
    data.times,
    data.static_covariates,
    data.outcome,
)
print(pooled.estimate)
print("Mean score loss before:", np.mean([item.loss_before for item in diagnostics]))
print("Mean score loss after:", np.mean([item.loss_after for item in diagnostics]))
