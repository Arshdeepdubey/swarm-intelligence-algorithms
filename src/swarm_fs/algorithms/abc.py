"""
Artificial Bee Colony (ABC) Optimization.

Implements the three bee roles and update equations from the IEEE paper
(Sec 4.2) and SDP report (Sec 3.3):

    Employed bee candidate generation:
        v(i,k) = x(i,k) + phi(i,k) * (x(i,k) - x(j,k))
    Onlooker bee (roulette-wheel) selection probability:
        P(i) = fit(i) / sum_j fit(j)
    Scout bee replacement of an exhausted food source:
        x(i,k) = lb(k) + phi(i,k) * (ub(k) - lb(k))

where ``phi`` ~ U(-1, 1) for employed/onlooker candidate generation and
``phi`` ~ U(0, 1) for scout re-initialization, ``j`` is a randomly chosen
neighboring food source (j != i), and ``k`` is a randomly chosen dimension.

Reference: Karaboga, D. (2005). "An idea based on honey bee swarm for
numerical optimization." Technical Report TR06, Erciyes University.

Note on the ``limit`` parameter: the source SDP report's parameter table
(Table 7) only documents population size (10), iterations (100), and the
search bounds ([0, 1]) -- it does not specify a scout "abandonment limit".
This implementation defaults to a standard literature heuristic
(``limit = population_size * dim``, floored at 10); see
``docs/documentation_review.md`` for this gap.
"""

from __future__ import annotations

import numpy as np

from .base import FitnessFn, OptimizationResult, SwarmOptimizerBase


class ArtificialBeeColony(SwarmOptimizerBase):
    def __init__(
        self,
        population_size: int = 10,
        max_iter: int = 100,
        lower: float = 0.0,
        upper: float = 1.0,
        seed: int | None = None,
        limit: int | None = None,
    ) -> None:
        super().__init__(population_size, max_iter, lower, upper, seed)
        self.limit = limit  # resolved to population_size * dim in optimize()

    def _fitness_to_fit(self, fitness: np.ndarray) -> np.ndarray:
        # Standard ABC transform from a minimization cost to a maximization
        # "fit" measure suitable for roulette-wheel selection.
        return np.where(fitness >= 0, 1.0 / (1.0 + fitness), 1.0 + np.abs(fitness))

    def _generate_candidate(self, positions: np.ndarray, i: int, dim: int) -> np.ndarray:
        candidates = [j for j in range(self.population_size) if j != i]
        j = self.rng.choice(candidates)
        k = self.rng.integers(0, dim)
        phi = self.rng.uniform(-1.0, 1.0)

        v = positions[i].copy()
        v[k] = positions[i, k] + phi * (positions[i, k] - positions[j, k])
        return np.clip(v, self.lower, self.upper)

    def optimize(self, fitness_fn: FitnessFn, dim: int) -> OptimizationResult:
        limit = self.limit if self.limit is not None else max(10, self.population_size * dim)

        positions = self._init_population(dim)
        fitness = np.empty(self.population_size)
        masks = np.zeros_like(positions, dtype=int)
        trials = np.zeros(self.population_size, dtype=int)

        for i in range(self.population_size):
            masks[i], fitness[i] = self._evaluate_mask(fitness_fn, positions[i])

        best_idx = int(np.argmin(fitness))
        best_position = positions[best_idx].copy()
        best_mask = masks[best_idx].copy()
        best_fitness = fitness[best_idx]

        convergence = np.empty(self.max_iter)

        for it in range(self.max_iter):
            # --- Employed bee phase ---
            for i in range(self.population_size):
                candidate = self._generate_candidate(positions, i, dim)
                cand_mask, cand_fit = self._evaluate_mask(fitness_fn, candidate)
                if cand_fit < fitness[i]:
                    positions[i] = candidate
                    fitness[i] = cand_fit
                    masks[i] = cand_mask
                    trials[i] = 0
                else:
                    trials[i] += 1

            # --- Onlooker bee phase (roulette-wheel over employed results) ---
            fit_values = self._fitness_to_fit(fitness)
            probs = fit_values / fit_values.sum()
            for _ in range(self.population_size):
                i = self.rng.choice(self.population_size, p=probs)
                candidate = self._generate_candidate(positions, i, dim)
                cand_mask, cand_fit = self._evaluate_mask(fitness_fn, candidate)
                if cand_fit < fitness[i]:
                    positions[i] = candidate
                    fitness[i] = cand_fit
                    masks[i] = cand_mask
                    trials[i] = 0
                else:
                    trials[i] += 1

            # --- Scout bee phase ---
            for i in range(self.population_size):
                if trials[i] >= limit:
                    phi = self.rng.uniform(0.0, 1.0, size=dim)
                    positions[i] = self.lower + phi * (self.upper - self.lower)
                    masks[i], fitness[i] = self._evaluate_mask(fitness_fn, positions[i])
                    trials[i] = 0

            gen_best = int(np.argmin(fitness))
            if fitness[gen_best] < best_fitness:
                best_fitness = fitness[gen_best]
                best_position = positions[gen_best].copy()
                best_mask = masks[gen_best].copy()

            convergence[it] = best_fitness

        return OptimizationResult(
            best_position=best_position,
            best_mask=best_mask,
            best_fitness=float(best_fitness),
            convergence_curve=convergence,
            n_evaluations=self.n_evaluations,
        )
