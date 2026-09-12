import numpy as np
import pytest

from swarm_fs.benchmarks import BENCHMARKS, ackley, penalized, sphere, step


def test_sphere_zero_at_origin():
    assert sphere(np.zeros(10)) == pytest.approx(0.0)


def test_sphere_positive_elsewhere():
    assert sphere(np.array([1.0, -2.0, 3.0])) == pytest.approx(1 + 4 + 9)


def test_step_zero_near_origin():
    assert step(np.zeros(5)) == pytest.approx(0.0)


def test_step_is_piecewise_constant():
    # Values within [-0.5, 0.5) all floor to the same bucket.
    assert step(np.array([0.1])) == step(np.array([0.4]))


def test_ackley_zero_at_origin():
    assert ackley(np.zeros(10)) == pytest.approx(0.0, abs=1e-8)


def test_ackley_positive_elsewhere():
    assert ackley(np.array([1.0, 1.0])) > 0


def test_penalized_zero_at_minus_one():
    dim = 8
    x = np.full(dim, -1.0)
    assert penalized(x) == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("name", list(BENCHMARKS.keys()))
def test_registered_benchmarks_have_expected_shape(name):
    spec = BENCHMARKS[name]
    lower, upper = spec["bounds"]
    assert lower < upper
    value = spec["fn"](np.zeros(5))
    assert np.isfinite(value)
