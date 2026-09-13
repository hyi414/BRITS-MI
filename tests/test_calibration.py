import numpy as np

from brits_mi.calibration import (
    AssociationCalibrationConfig,
    calibrate_completed_trajectory,
)
from brits_mi.simulation import TRUE_BETA, SimulationConfig, simulate_clinical_association


def test_score_calibration_locks_observed_cells_and_does_not_increase_loss():
    data = simulate_clinical_association(
        SimulationConfig(n_subjects=80, target_missing=0.30, seed=19)
    )
    completed = np.where(data.observed_mask, data.true_values, 0.0)
    coefficients = np.array(list(TRUE_BETA.values()), dtype=float)
    calibrated, diagnostic = calibrate_completed_trajectory(
        completed,
        data.observed_values,
        data.observed_mask,
        data.times,
        data.static_covariates,
        data.outcome,
        coefficients,
        intercept=data.config.outcome_intercept,
        config=AssociationCalibrationConfig(step_size=0.02, n_steps=1),
    )
    assert np.allclose(
        calibrated[data.observed_mask], data.true_values[data.observed_mask]
    )
    assert diagnostic.loss_after <= diagnostic.loss_before + 1e-8
