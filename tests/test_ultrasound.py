import numpy as np

from brits_mi.ultrasound import simulate_ultrasound_context


def test_ultrasound_renderer_and_features():
    images, features = simulate_ultrasound_context(
        severity=np.array([-0.5, 0.2, 1.1]),
        activity=np.array([0.1, 0.8, 1.2]),
        stages=np.array([0, 1, 2]),
        seed=4,
    )
    assert images.shape == (3, 1, 24, 24)
    assert features.shape == (3, 17)
    assert np.isfinite(images).all()
    assert np.isfinite(features).all()
