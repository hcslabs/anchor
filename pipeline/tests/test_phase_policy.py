import numpy as np
import pytest

from anchor.policy.phase_policy import build_phase_features


def test_phase_features_have_stable_contract():
    features = build_phase_features(np.array([0.0, 0.5, 1.0], dtype=np.float32))
    assert features.shape == (3, 9)
    assert np.isfinite(features).all()
    assert features[0, 0] == pytest.approx(0.0)
    assert features[-1, 0] == pytest.approx(1.0)


def test_phase_features_reject_out_of_range():
    with pytest.raises(ValueError):
        build_phase_features(np.array([-0.1, 0.5]))

