import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from brits_mi import recurrent_image
from brits_mi.simulation import SimulationConfig, simulate_clinical_association


def test_image_comparator_has_no_outcome_column(monkeypatch):
    captured = {}

    def fake(*args, **kwargs):
        captured.update(kwargs)
        return [args[0]], {}

    monkeypatch.setattr(recurrent_image, "r_impute", fake)
    d = simulate_clinical_association(SimulationConfig(n_subjects=80, seed=33))
    image = np.ones((80, 17))
    recurrent_image.impute_image_comparator(d.observed_values, d.observed_mask,
        d.times, d.static_covariates, image, np.full(80, 6), "MICE", 20, 11)
    assert captured.get("outcome") is None
    np.testing.assert_array_equal(captured["context"], image)
    assert captured["m"] == 20


def test_equal_head_fit_budget_and_test_label_invariance(monkeypatch):
    torch.set_num_threads(1)
    d = simulate_clinical_association(SimulationConfig(n_subjects=80, seed=33))
    times = np.broadcast_to(d.times, (80, 6))
    labels = np.arange(80) % 3
    args = ([d.true_values], times, d.static_covariates,
            np.zeros((80, 17)), np.full(80, 6))
    calls = []
    original = torch.optim.AdamW

    def record(*a, **kw):
        calls.append(1)
        return original(*a, **kw)

    monkeypatch.setattr(torch.optim, "AdamW", record)
    train, test = np.arange(60), np.arange(60, 80)
    a = recurrent_image.fusion_probabilities(*args, labels, train, test, 51,
                                           epochs=2, n_fits=3)
    assert len(calls) == 3
    labels[test] = -1
    b = recurrent_image.fusion_probabilities(*args, labels, train, test, 51,
                                           epochs=2, n_fits=3)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a.sum(1), 1, atol=1e-6)
    with pytest.raises(ValueError, match="positive integer"):
        recurrent_image.fusion_probabilities(*args, labels, train, test, 51, n_fits=0)
    with pytest.raises(ValueError, match="one imputation per classifier fit"):
        recurrent_image.fusion_probabilities([d.true_values] * 2, *args[1:],
            labels, train, test, 51, n_fits=3)


def test_duplicate_runner_is_rejected(tmp_path):
    path = Path(__file__).resolve().parents[1] / "simulation/unified/run_benchmark.py"
    spec = importlib.util.spec_from_file_location("unified_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    with runner.exclusive_run(tmp_path):
        with pytest.raises(RuntimeError, match="already holds this output lock"):
            with runner.exclusive_run(tmp_path):
                pass
    with runner.exclusive_run(tmp_path):
        pass
