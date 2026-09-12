"""
Cuckoo Search Algorithm (CSA).

Implements the Levy-flight generation rule from the IEEE paper (Sec 4.4)
and SDP report (Sec 3.3):

    x(t+1) = x(t) + s * E(t)

where the SDP report specifies ``E(t)`` is drawn from a standard normal
distribution for the random-walk formulation, and separately references
Levy flights governed by a survivor function with fractal dimension D (a
particular case of the Pareto distribution). This implementation uses the
standard Mantegna algorithm to sample Levy-stable steps (the
textbook-standard way to realize "Levy flights" in CSA, per Yang & Deb's
original paper), combined with a per-nest step-size scale ``alpha``.

Each generation:
    1. Generate a new solution for a randomly chosen nest via a Levy
       flight and greedily replace it if better (elitism, "keeps the
       best").
    2. Abandon a fraction ``pa`` of the worst nests, replacing them with
       fresh random solutions (approximating the paper's "host bird
       discovers the alien egg" step).

Reference: Yang, X.-S., & Deb, S. (2009). "Cuckoo search via Levy
flights." World Congress on Nature & Biologically Inspired Computing.

Note: ``pa`` (nest-abandonment probability) and the Levy step-size scale
are not specified in the SDP report's parameter table (Table 7 lists only
population size, iterations, and bounds), so this implementation uses the
values from Yang & Deb's original paper (``pa = 0.25``) as a documented
default -- see ``docs/documentation_review.md``.
"""

from __future__ import annotations

import numpy as np

from .base import FitnessFn, OptimizationResult, SwarmOptimizerBase
from ..utils import levy_flight


class CuckooSearch(SwarmOptimizerBase):
    def __init__(
        self,
        population_size: int = 10,
        max_iter: int = 100,
        lower: float = 0.0,
        upper: float = 1.0,
        seed: int | None = None,
        pa: float = 0.25,
        step_scale: float = 0.1,
    ) -> None:
        super().__init__(population_size, max_iter, lower, upper, seed)
        self.pa = pa
        self.step_scale = step_scale

    def optimize(self, fitness_fn: FitnessFn, dim: int) -> OptimizationResult:
        nests = self._init_population(dim)
        fitness = np.empty(self.population_size)
        masks = np.zeros_like(nests, dtype=int)

        for i in range(self.population_size):
            masks[i], fitness[i] = self._evaluate_mask(fitness_fn, nests[i])

        best_idx = int(np.argmin(fitness))
        best_position = nests[best_idx].copy()
        best_mask = masks[best_idx].copy()
        best_fitness = fitness[best_idx]

        convergence = np.empty(self.max_iter)
        span = self.upper - self.lower

        for it in range(self.max_iter):
            # --- Get a cuckoo via a Levy flight, evaluate against a random nest ---
            for _ in range(self.population_size):
                i = self.rng.integers(0, self.population_size)
                step = levy_flight(dim, self.rng) * self.step_scale * span
                new_pos = np.clip(nests[i] + step, self.lower, self.upper)
                new_mask, new_fit = self._evaluate_mask(fitness_fn, new_pos)

                j = self.rng.integers(0, self.population_size)
                if new_fit < fitness[j]:
                    nests[j] = new_pos
                    fitness[j] = new_fit
                    masks[j] = new_mask

            # --- Abandon a fraction pa of the worst nests ---
            n_abandon = int(self.pa * self.population_size)
            if n_abandon > 0:
                worst_idx = np.argsort(fitness)[-n_abandon:]
                for idx in worst_idx:
                    nests[idx] = self.rng.uniform(self.lower, self.upper, size=dim)
                    masks[idx], fitness[idx] = self._evaluate_mask(fitness_fn, nests[idx])

            gen_best = int(np.argmin(fitness))
            if fitness[gen_best] < best_fitness:
                best_fitness = fitness[gen_best]
                best_position = nests[gen_best].copy()
                best_mask = masks[gen_best].copy()

            convergence[it] = best_fitness

        return OptimizationResult(
            best_position=best_position,
            best_mask=best_mask,
            best_fitness=float(best_fitness),
            convergence_curve=convergence,
            n_evaluations=self.n_evaluations,
        )
