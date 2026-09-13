import numpy as np

from brits_mi import BRITSMultipleImputer, TrainingConfig
from brits_mi.simulation import SimulationConfig, simulate_clinical_association


def test_multiple_completion_locks_measured_cells():
    data = simulate_clinical_association(
        SimulationConfig(n_subjects=80, target_missing=0.30, seed=9)
    )
    imputer = BRITSMultipleImputer(
        TrainingConfig(
            epochs=2,
            hidden_size=10,
            batch_size=32,
            variance_factor=0.50,
            seed=9,
            device="cpu",
        )
    )
    imputer.fit(
        data.observed_values,
        data.observed_mask,
        data.times,
        data.static_covariates,
        data.outcome,
    )
    draws = imputer.sample(2, seed=99)
    assert np.allclose(draws[0][data.observed_mask], data.true_values[data.observed_mask])
    assert np.allclose(draws[1][data.observed_mask], data.true_values[data.observed_mask])
    missing = ~data.observed_mask
    assert np.any(np.abs(draws[0][missing] - draws[1][missing]) > 1e-8)

