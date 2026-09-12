"""
Particle Swarm Optimization (PSO).

Implements the velocity/position update equations exactly as given in the
IEEE paper (Sec 4.1) and the SDP report (Sec 3.3, Fig. 3):

    v_i^k = w * v_i^k + c1*r1*(pbest_i^k - x_i^k) + c2*r2*(gbest^k - x_i^k)
    x_i^{k+1} = x_i^k + v_i^{k+1}

where ``w`` is the inertia weight, ``c1``/``c2`` are the cognitive/social
acceleration constants, and ``r1``/``r2`` ~ U(0, 1) are independent random
draws per dimension, per particle, per iteration.

Reference: Kennedy & Eberhart, "Particle Swarm Optimization", 1995.
"""

from __future__ import annotations

import numpy as np

from .base import FitnessFn, OptimizationResult, SwarmOptimizerBase


class ParticleSwarmOptimization(SwarmOptimizerBase):
    def __init__(
        self,
        population_size: int = 10,
        max_iter: int = 100,
        lower: float = 0.0,
        upper: float = 1.0,
        seed: int | None = None,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        v_max: float | None = None,
    ) -> None:
        super().__init__(population_size, max_iter, lower, upper, seed)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max if v_max is not None else (upper - lower)

    def optimize(self, fitness_fn: FitnessFn, dim: int) -> OptimizationResult:
        positions = self._init_population(dim)
        velocities = self.rng.uniform(-self.v_max, self.v_max, size=(self.population_size, dim))

        pbest_positions = positions.copy()
        pbest_masks = np.zeros_like(positions, dtype=int)
        pbest_fitness = np.full(self.population_size, np.inf)

        for i in range(self.population_size):
            mask, fit = self._evaluate_mask(fitness_fn, positions[i])
            pbest_fitness[i] = fit
            pbest_masks[i] = mask

        gbest_idx = int(np.argmin(pbest_fitness))
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_mask = pbest_masks[gbest_idx].copy()
        gbest_fitness = pbest_fitness[gbest_idx]

        convergence = np.empty(self.max_iter)

        for it in range(self.max_iter):
            r1 = self.rng.random(size=(self.population_size, dim))
            r2 = self.rng.random(size=(self.population_size, dim))

            velocities = (
                self.w * velocities
                + self.c1 * r1 * (pbest_positions - positions)
                + self.c2 * r2 * (gbest_position - positions)
            )
            velocities = np.clip(velocities, -self.v_max, self.v_max)
            positions = np.clip(positions + velocities, self.lower, self.upper)

            for i in range(self.population_size):
                mask, fit = self._evaluate_mask(fitness_fn, positions[i])
                if fit < pbest_fitness[i]:
                    pbest_fitness[i] = fit
                    pbest_positions[i] = positions[i].copy()
                    pbest_masks[i] = mask
                if fit < gbest_fitness:
                    gbest_fitness = fit
                    gbest_position = positions[i].copy()
                    gbest_mask = mask.copy()

            convergence[it] = gbest_fitness

        return OptimizationResult(
            best_position=gbest_position,
            best_mask=gbest_mask,
            best_fitness=float(gbest_fitness),
            convergence_curve=convergence,
            n_evaluations=self.n_evaluations,
        )
