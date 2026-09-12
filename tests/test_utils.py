import numpy as np
import pytest

from swarm_fs.utils import binarize, clip, ensure_at_least_one_feature, levy_flight, sigmoid


def test_sigmoid_matches_definition():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    expected = 1.0 / (1.0 + np.exp(-x))
    np.testing.assert_allclose(sigmoid(x), expected, atol=1e-10)


def test_sigmoid_at_zero_is_half():
    assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)


def test_sigmoid_no_overflow_on_extreme_values():
    x = np.array([-1000.0, 1000.0])
    out = sigmoid(x)
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(0.0, abs=1e-6)
    assert out[1] == pytest.approx(1.0, abs=1e-6)


def test_sigmoid_output_range():
    rng = np.random.default_rng(0)
    x = rng.uniform(-50, 50, size=1000)
    out = sigmoid(x)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_binarize_output_is_binary():
    rng = np.random.default_rng(1)
    position = rng.uniform(-5, 5, size=20)
    mask = binarize(position, rng)
    assert set(np.unique(mask)).issubset({0, 1})
    assert mask.shape == position.shape


def test_binarize_large_positive_tends_to_select():
    rng = np.random.default_rng(2)
    position = np.full(200, 10.0)  # sigmoid(10) ~ 0.99995, almost always > threshold
    mask = binarize(position, rng)
    assert mask.mean() > 0.95


def test_binarize_large_negative_tends_to_reject():
    rng = np.random.default_rng(3)
    position = np.full(200, -10.0)
    mask = binarize(position, rng)
    assert mask.mean() < 0.05


def test_ensure_at_least_one_feature_fixes_empty_mask():
    rng = np.random.default_rng(4)
    mask = np.zeros(10, dtype=int)
    fixed = ensure_at_least_one_feature(mask, rng)
    assert fixed.sum() == 1


def test_ensure_at_least_one_feature_leaves_nonempty_mask_untouched():
    rng = np.random.default_rng(5)
    mask = np.array([0, 1, 0, 1])
    fixed = ensure_at_least_one_feature(mask, rng)
    np.testing.assert_array_equal(fixed, mask)


def test_clip_respects_bounds():
    x = np.array([-5.0, 0.5, 5.0])
    out = clip(x, 0.0, 1.0)
    np.testing.assert_array_equal(out, [0.0, 0.5, 1.0])


def test_levy_flight_shape_and_finiteness():
    rng = np.random.default_rng(6)
    step = levy_flight(dim=15, rng=rng)
    assert step.shape == (15,)
    assert np.all(np.isfinite(step))
