import numpy as np
import pandas as pd
import pytest
import torch

from brits_mi.analysis import validated_logistic_fit, trajectory_features, barnard_rubin_df
from brits_mi.benchmark import pool_designs
from brits_mi.comparators import comparator_frame
from brits_mi.simulation import simulate_clinical_association, SimulationConfig
from brits_mi.unified import UnifiedBRITSMI, UnifiedConfig, RecurrentNetwork, fusion_features


def dataset():
    return simulate_clinical_association(SimulationConfig(n_subjects=80, seed=33))


def test_invalid_glms_are_rejected():
    x = np.tile([0., 1.], 40)
    with pytest.raises(ValueError):
        validated_logistic_fit(x, pd.DataFrame({"diabetes": x}))
    with pytest.raises(ValueError, match="rank"):
        validated_logistic_fit(x, pd.DataFrame({"constant": np.zeros(80)}))


def test_quasi_separation_is_rejected_even_with_overlap():
    x = np.repeat([0., 1.], 40)
    y = np.concatenate([np.tile([0., 1.], 20), np.ones(40)])
    with pytest.raises(ValueError, match="quasi-separated"):
        validated_logistic_fit(y, pd.DataFrame({"x": x}))


def test_subject_specific_times_are_used():
    d = dataset()
    times = np.broadcast_to(d.times, (80, 6)).copy()
    times[0] *= 2
    result = trajectory_features(d.true_values, times)
    shared = trajectory_features(d.true_values, d.times)
    assert result.loc[0, "slope_ast"] == pytest.approx(shared.loc[0, "slope_ast"] / 2)


def test_pooling_uses_one_reference_sd():
    rng = np.random.default_rng(8)
    x = rng.normal(size=200)
    y = rng.binomial(1, 1 / (1 + np.exp(-.6 * x)))
    a = pool_designs([pd.DataFrame({"x": x})], y, [2.], [.6 * 2])
    b = pool_designs([pd.DataFrame({"x": x * 10})], y, [20.], [.6 * 2])
    np.testing.assert_allclose(a.estimate, b.estimate)
    np.testing.assert_allclose(a.standard_error, b.standard_error)


def test_barnard_rubin_matches_mice_documented_example():
    estimates = np.array([-1.5457, -1.428])
    variances = np.square([.9723, 1.041])
    df = barnard_rubin_df(variances.mean(), estimates.var(ddof=1), 2, 23)
    assert df == pytest.approx(20.97025, abs=1e-5)


def test_comparators_receive_outcome_and_no_hidden_values():
    d = dataset()
    f, targets = comparator_frame(d.true_values, d.observed_mask, d.times,
                                  d.static_covariates, d.outcome)
    assert np.array_equal(f.outcome, d.outcome)
    assert np.isnan(f[targets].to_numpy()[~d.observed_mask.reshape(80, -1)]).all()


def test_unified_mask_boundary_and_observation_lock():
    torch.set_num_threads(1)
    d = dataset()
    config = UnifiedConfig(epochs=2, hidden_size=8, seed=12)
    a = UnifiedBRITSMI(config).fit(d.observed_values, d.observed_mask, d.times,
                                  d.static_covariates, d.outcome)
    poisoned = np.where(d.observed_mask, d.true_values, 1e10)
    b = UnifiedBRITSMI(config).fit(poisoned, d.observed_mask, d.times,
                                  d.static_covariates, d.outcome)
    aa, bb = a.sample(2, seed=10), b.sample(2, seed=10)
    np.testing.assert_allclose(aa, bb)
    for draw in aa:
        np.testing.assert_array_equal(draw[d.observed_mask], d.true_values[d.observed_mask])
    assert not np.allclose(aa[0][~d.observed_mask], aa[1][~d.observed_mask])
    for covariance in a.residual_covariances_:
        np.testing.assert_array_equal(covariance, covariance.T)
        assert covariance.dtype == np.float64
        assert np.linalg.eigvalsh(covariance).min() > 0
    with np.errstate(all="raise"):
        a.sample(2, seed=10, residual_mode="correlated")
    with pytest.raises(ValueError, match="observed outcomes"):
        a.sample(1, values=d.observed_values[:12], mask=d.observed_mask[:12],
                 times=d.times, static=d.static_covariates[:12])


def test_image_gradient_reaches_both_recurrent_paths():
    torch.manual_seed(77)
    n = 12
    net = RecurrentNetwork(3, 3, 2, 3, 8, 27, False, True)
    y = torch.randn(n, 6, 3)
    mask = (torch.rand_like(y) > .4).float()
    time = torch.linspace(0, 1, 6).repeat(n, 1)
    static = torch.randn(n, 3)
    context = torch.randn(n, 2, requires_grad=True)
    mean, completed, _, _ = net(y * mask, mask, time, static, context, torch.zeros(n, 0))
    f = fusion_features(completed, time, static)
    loss = torch.nn.functional.cross_entropy(net.head(torch.cat([f, context], 1)), torch.arange(n) % 3)
    loss.backward()
    assert net.forward_rits.history_head.weight.grad.abs().sum() > 0
    assert net.backward_rits.history_head.weight.grad.abs().sum() > 0
    assert context.grad.abs().sum() > 0
    with torch.no_grad():
        shifted = net(y * mask, mask, time, static, context + 1, torch.zeros(n, 0))[0]
    assert not torch.allclose(mean[mask == 0], shifted[mask == 0])


def test_prediction_does_not_accept_query_labels():
    torch.set_num_threads(1)
    d = dataset()
    x = UnifiedBRITSMI(UnifiedConfig(epochs=1, hidden_size=8), task="prediction")
    x.fit(d.observed_values, d.observed_mask, d.times, d.static_covariates, d.outcome)
    with pytest.raises(ValueError, match="must not be supplied"):
        x.sample(1, values=d.observed_values, mask=d.observed_mask, times=d.times,
                 static=d.static_covariates, outcome=d.outcome)
    with pytest.raises(ValueError, match="must not be supplied"):
        x.sample(1, outcome=d.outcome)


def test_padding_does_not_change_valid_predictions():
    torch.manual_seed(2)
    net = RecurrentNetwork(3, 3, 0, 2, 8, 13, False, False)
    y = torch.randn(4, 6, 3)
    mask = torch.ones_like(y)
    mask[:, 4:] = 0
    times = torch.arange(6).float().repeat(4, 1)
    static = torch.ones(4, 3)
    a = net(y, mask, times, static, torch.zeros(4, 0), torch.zeros(4, 0), torch.full((4,), 4))[0][:, :4]
    b = net(y[:, :4], mask[:, :4], times[:, :4], static, torch.zeros(4, 0), torch.zeros(4, 0))[0]
    torch.testing.assert_close(a, b)


def test_time_scaling_uses_last_real_visit():
    torch.set_num_threads(1)
    d = dataset()
    mask = d.observed_mask.copy()
    mask[:, 4:] = False
    times = np.tile([0., 1., 2., 3., 100., 200.], (80, 1))
    lengths = np.full(80, 4)
    model = UnifiedBRITSMI(UnifiedConfig(epochs=1, hidden_size=8), task="prediction",
                          feature_function=fusion_features)
    model.fit(d.observed_values, mask, times, d.static_covariates, d.outcome, lengths=lengths)
    assert model.time_scale_ == 3.
    bad_mask = mask.copy()
    bad_mask[:, 5] = True
    with pytest.raises(ValueError, match="padding"):
        model.fit(d.true_values, bad_mask, times, d.static_covariates, d.outcome, lengths=lengths)


def test_fusion_head_does_not_read_test_labels():
    from brits_mi.recurrent_image import fusion_probabilities
    torch.set_num_threads(1)
    d = dataset()
    rng = np.random.default_rng(31)
    labels = np.arange(80) % 3
    times = np.broadcast_to(d.times, d.true_values.shape[:2])
    image = rng.normal(size=(80, 2))
    train, test = np.arange(60), np.arange(60, 80)
    args = ([d.true_values], times, d.static_covariates, image, np.full(80, 6))
    a = fusion_probabilities(*args, labels, train, test, 7, epochs=3)
    labels[test] = -1
    b = fusion_probabilities(*args, labels, train, test, 7, epochs=3)
    np.testing.assert_array_equal(a, b)
