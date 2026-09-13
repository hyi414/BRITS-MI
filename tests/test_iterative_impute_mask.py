"""Regression tests for mask-based missingness in the image benchmark baseline."""

from types import SimpleNamespace

import numpy as np

from longitudinal_sim.baselines import iterative_impute


def sample_data():
    rng = np.random.default_rng(491)
    x = rng.normal(size=(50, 2))
    y = 3 + x[:, 0, None, None] + rng.normal(scale=0.2, size=(50, 4, 2))
    mask = (rng.random(y.shape) > 0.3).astype(float)
    y[0, 0, 0] = 0
    mask[0, 0, 0] = 1
    return SimpleNamespace(
        y_obs=np.where(mask == 1, y, 0), mask=mask, static=x, x=np.zeros((50, 4, 1))
    )


def test_missing_placeholders_are_imputed_and_observed_zeros_are_preserved():
    data = sample_data()
    completed = iterative_impute(data, np.arange(35))
    assert completed.shape == data.y_obs.shape
    assert np.array_equal(completed[data.mask == 1], data.y_obs[data.mask == 1])
    assert completed[0, 0, 0] == 0
    assert np.any(completed[data.mask == 0] != 0)
    assert np.isfinite(completed).all()


def test_empty_training_feature_retains_dimension():
    data = sample_data()
    data.mask[:35, 0, 1] = 0
    data.y_obs[:35, 0, 1] = 0
    completed = iterative_impute(data, np.arange(35))
    assert completed.shape == data.y_obs.shape
    assert np.isfinite(completed).all()
    assert np.array_equal(completed[data.mask == 1], data.y_obs[data.mask == 1])


def test_test_targets_cannot_change_training_completions():
    a = sample_data()
    before = iterative_impute(a, np.arange(35))
    a.y_obs[35:] = np.where(a.mask[35:] == 1, 1000, 0)
    after = iterative_impute(a, np.arange(35))
    assert np.array_equal(before[:35], after[:35])
