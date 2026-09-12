"""
Shared result type and a tiny base class for the four SI optimizers.

Every optimizer in this package works over a real-valued search space of
``dim`` dimensions bounded to ``[lower, upper]`` (Table 7 in the SDP
report: lower bound 0, upper bound 1, dimensions "depending on the
dataset" -- i.e. the number of candidate features). A continuous position
is converted to a binary feature mask via :func:`swarm_fs.utils.binarize`
before being scored, exactly as described in Section 3.2.2 of the IEEE
paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..utils import binarize, ensure_at_least_one_feature

FitnessFn = Callable[[np.ndarray], float]


@dataclass
class OptimizationResult:
    """Outcome of running one SI optimizer to completion."""

    best_position: np.ndarray  # continuous (or pheromone-derived) position
    best_mask: np.ndarray  # binary feature mask
    best_fitness: float
    convergence_curve: np.ndarray  # best fitness per iteration (non-increasing)
    n_evaluations: int


class SwarmOptimizerBase:
    """Common bookkeeping (bounds, seeding, mask caching) for SI optimizers.

    Subclasses implement :meth:`optimize`. This base class is intentionally
    small: PSO, ABC, and CSA share a continuous-position/binary-mask
    representation and use the helpers below; ACO instead builds masks
    directly from pheromone (see :mod:`swarm_fs.algorithms.aco`) and only
    reuses ``_evaluate_mask``.
    """

    def __init__(
        self,
        population_size: int = 10,
        max_iter: int = 100,
        lower: float = 0.0,
        upper: float = 1.0,
        seed: int | None = None,
    ) -> None:
        # population_size=10, max_iter=100 match Table 7 ("Parameters
        # Used") in the SDP report.
        self.population_size = population_size
        self.max_iter = max_iter
        self.lower = lower
        self.upper = upper
        self.rng = np.random.default_rng(seed)
        self.n_evaluations = 0

    def _evaluate_mask(self, fitness_fn: FitnessFn, position: np.ndarray) -> tuple[np.ndarray, float]:
        mask = binarize(position, self.rng)
        mask = ensure_at_least_one_feature(mask, self.rng)
        self.n_evaluations += 1
        return mask, fitness_fn(mask)

    def _init_population(self, dim: int) -> np.ndarray:
        return self.rng.uniform(self.lower, self.upper, size=(self.population_size, dim))
