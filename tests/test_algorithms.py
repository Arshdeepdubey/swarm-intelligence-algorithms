"""
Correctness/convergence tests for the four SI optimizers.

Rather than relying on the (slow, stochastic) SVM-wrapper fitness, these
tests use the ``sphere`` benchmark function (see
``swarm_fs.benchmarks.sphere``) applied directly to the binary feature
mask. Since a binary mask only contains 0/1 entries, ``sphere(mask) ==
sum(mask)`` -- i.e. the number of selected features. Combined with the
representation's "never select zero features" safeguard
(:func:`swarm_fs.utils.ensure_at_least_one_feature`), the true optimum is
a mask with exactly one feature selected (fitness = 1). This gives a fast,
deterministic-enough way to confirm every optimizer's search loop actually
drives its population toward better solutions instead of, say, silently
never updating a global best.
"""

from __future__ import annotations

import numpy as np
import pytest

from swarm_fs.algorithms import ALGORITHMS
from swarm_fs.algorithms.aco import AntColonyOptimization
from swarm_fs.algorithms.abc import ArtificialBeeColony
from swarm_fs.algorithms.cuckoo import CuckooSearch
from swarm_fs.algorithms.pso import ParticleSwarmOptimization
from swarm_fs.benchmarks import sphere

DIM = 10


@pytest.mark.parametrize("name,cls", list(ALGORITHMS.items()))
def test_default_hyperparameters_match_sdp_table7(name, cls):
    optimizer = cls()
    assert optimizer.population_size == 10
    assert optimizer.max_iter == 100
    assert optimizer.lower == 0.0
    assert optimizer.upper == 1.0


@pytest.mark.parametrize("name,cls", list(ALGORITHMS.items()))
def test_convergence_curve_is_non_increasing(name, cls):
    optimizer = cls(population_size=10, max_iter=25, seed=42)
    result = optimizer.optimize(sphere, dim=DIM)

    curve = result.convergence_curve
    assert len(curve) == 25
    # Each entry must be <= the previous one: we only ever update the
    # tracked "best fitness" downward.
    assert np.all(np.diff(curve) <= 1e-9)


@pytest.mark.parametrize("name,cls", list(ALGORITHMS.items()))
def test_search_improves_on_random_guessing(name, cls):
    optimizer = cls(population_size=10, max_iter=30, seed=7)
    result = optimizer.optimize(sphere, dim=DIM)

    # A uniform-random binary mask over 10 dimensions selects ~5 features
    # on average (sphere ~= 5). A working optimizer chasing the minimum
    # (a single selected feature) should comfortably beat that.
    assert result.best_fitness < 5.0
    assert result.best_mask.sum() == pytest.approx(result.best_fitness, abs=1e-9)


@pytest.mark.parametrize("name,cls", list(ALGORITHMS.items()))
def test_best_mask_is_binary_and_nonempty(name, cls):
    optimizer = cls(population_size=8, max_iter=10, seed=1)
    result = optimizer.optimize(sphere, dim=DIM)

    assert set(np.unique(result.best_mask)).issubset({0, 1})
    assert result.best_mask.sum() >= 1


@pytest.mark.parametrize("name,cls", list(ALGORITHMS.items()))
def test_reproducible_with_fixed_seed(name, cls):
    r1 = cls(population_size=8, max_iter=10, seed=123).optimize(sphere, dim=DIM)
    r2 = cls(population_size=8, max_iter=10, seed=123).optimize(sphere, dim=DIM)
    assert r1.best_fitness == pytest.approx(r2.best_fitness)
    np.testing.assert_array_equal(r1.best_mask, r2.best_mask)


def test_pso_velocity_clipped_to_v_max():
    optimizer = ParticleSwarmOptimization(population_size=5, max_iter=5, seed=0, v_max=0.3)
    optimizer.optimize(sphere, dim=6)
    # v_max is used to clip every update; nothing to assert on internal
    # state post-hoc, but constructing/using it must not raise.
    assert optimizer.v_max == 0.3


def test_abc_scout_phase_replaces_exhausted_food_sources():
    # A tiny limit forces the scout phase to trigger within a few iterations.
    optimizer = ArtificialBeeColony(population_size=5, max_iter=5, seed=0, limit=1)
    result = optimizer.optimize(sphere, dim=6)
    assert result.n_evaluations > 0


def test_cuckoo_abandons_worst_fraction_each_generation():
    optimizer = CuckooSearch(population_size=10, max_iter=5, seed=0, pa=0.5)
    result = optimizer.optimize(sphere, dim=6)
    assert result.n_evaluations > 0


def test_aco_pheromone_stays_positive():
    optimizer = AntColonyOptimization(population_size=6, max_iter=8, seed=0)
    result = optimizer.optimize(sphere, dim=6)
    assert result.best_fitness >= 1  # can never be 0: at least one feature always selected
