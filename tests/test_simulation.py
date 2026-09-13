import numpy as np

from brits_mi.simulation import SimulationConfig, simulate_clinical_association


def test_simulation_is_reproducible_and_has_requested_shapes():
    config = SimulationConfig(n_subjects=80, target_missing=0.40, seed=17)
    first = simulate_clinical_association(config)
    second = simulate_clinical_association(config)
    assert first.true_values.shape == (80, 6, 3)
    assert first.static_covariates.shape == (80, 3)
    assert np.array_equal(first.observed_mask, second.observed_mask)
    assert np.allclose(first.true_values, second.true_values)
    assert 0.25 < (~first.observed_mask[:, -3:, :]).mean() < 0.55

