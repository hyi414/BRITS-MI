import numpy as np
import pytest

from brits_mi._image_reference import trajectory_motif_features
from brits_mi.image_fusion import BRITSMIImage, ImageFusionConfig


@pytest.fixture
def sample():
    rng = np.random.default_rng(182)
    n, t = 48, 4
    y = rng.normal(size=(n, t, 3))
    mask = rng.random(y.shape) > 0.3
    mask[:, 0, 0] = False
    y[mask & (y > 1)] = 0
    return (
        np.where(mask, y, np.nan),
        mask,
        np.tile(np.arange(t), (n, 1)),
        np.full(n, t),
        rng.normal(size=(n, 2)),
        rng.random((n, 1, 24, 24)),
    ), np.arange(n) % 3


def test_completion_is_label_free_and_measured_values_are_fixed(sample):
    args, labels = sample
    model = BRITSMIImage(
        ImageFusionConfig(cell_trees=4, fusion_trees=4, min_cell_training=4, prototype_alpha=0.2)
    ).fit(*args, labels)
    completed = model.complete(*args)
    np.testing.assert_array_equal(completed[args[1]], args[0][args[1]])
    assert np.isfinite(completed).all()
    assert trajectory_motif_features(completed, args[3]).shape == (48, 34)
    probabilities = model.predict_proba(*args)
    assert probabilities.shape == (48, 3)
    np.testing.assert_allclose(probabilities.sum(1), 1)


def test_new_patient_predictions_do_not_depend_on_query_cohort(sample):
    args, labels = sample
    model = BRITSMIImage(ImageFusionConfig(cell_trees=4, fusion_trees=4, min_cell_training=4)).fit(
        *args, labels
    )
    alone = model.predict_proba(*(a[:1] for a in args))
    np.testing.assert_allclose(alone, model.predict_proba(*args)[:1])


def test_repeatability_and_input_validation(sample):
    args, labels = sample
    config = ImageFusionConfig(cell_trees=3, fusion_trees=3, min_cell_training=4)
    a = BRITSMIImage(config).fit(*args, labels)
    b = BRITSMIImage(config).fit(*args, labels)
    np.testing.assert_array_equal(a.complete(*args), b.complete(*args))
    with pytest.raises(ValueError):
        ImageFusionConfig(prototype_alpha=1.1)
    with pytest.raises(ValueError):
        BRITSMIImage(config).fit(*args, np.zeros(48))
    with pytest.raises(RuntimeError):
        BRITSMIImage(config).complete(*args)
