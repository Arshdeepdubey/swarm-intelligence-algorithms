"""
(Binary) Ant Colony Optimization for feature selection.

Classical ACO builds *paths* over a graph using pheromone trails (Sec 4.3
of the IEEE paper, Sec 3.3 of the SDP report):

    Edge selection probability:
        p^k(x,y) = [tau(x,y)]^alpha * [eta(x,y)]^beta
                   / sum_{z in allowed(x)} [tau(x,z)]^alpha * [eta(x,z)]^beta

    Global pheromone update:
        tau(x,y) <- (1-rho)*tau(x,y) + sum_{k=1}^{m} delta_tau_k(x,y)
        delta_tau_k(x,y) = Q / L_k   if ant k used edge (x,y)
                          = 0        otherwise

Feature selection has no natural graph of "cities" to tour, so -- following
the standard adaptation of ACO to feature selection used throughout the
literature the survey paper cites (binary/pheromone-per-feature ACO, e.g.
Wan et al. 2016, ref [4] in the IEEE paper) -- this implementation treats
each feature ``j`` as a binary decision with two "edges" an ant can take:
select it (state 1) or skip it (state 0). Each edge carries its own
pheromone value ``tau[j, state]``. An ant constructs a full feature mask by
independently sampling a state for every feature using exactly the edge
selection formula above (with a neutral heuristic eta=1, since the report
does not define a domain-specific heuristic). Pheromone is then updated
globally with the same evaporation + deposit rule, using each ant's tour
cost ``L_k`` = the SVM-wrapper fitness (lower fitness => more pheromone
deposited via Q/L_k, so better feature masks reinforce their edges).

Note: neither ``alpha``/``beta`` (pheromone/heuristic influence) nor
``rho`` (evaporation rate) nor ``Q`` are specified in the SDP report's
parameter table (Table 7 only lists population size, iterations, and
bounds). This implementation uses commonly cited ACO defaults
(alpha=1, beta=1, rho=0.5, Q=1.0) -- see
``docs/documentation_review.md`` for this gap.
"""

from __future__ import annotations

import numpy as np

from .base import FitnessFn, OptimizationResult, SwarmOptimizerBase

_EPS = 1e-12


class AntColonyOptimization(SwarmOptimizerBase):
    def __init__(
        self,
        population_size: int = 10,
        max_iter: int = 100,
        lower: float = 0.0,
        upper: float = 1.0,
        seed: int | None = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        rho: float = 0.5,
        Q: float = 1.0,
    ) -> None:
        # lower/upper are accepted for interface parity with the other
        # optimizers but are not used: ACO here works natively in {0, 1}.
        super().__init__(population_size, max_iter, lower, upper, seed)
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q

    def _construct_mask(self, tau: np.ndarray, eta: np.ndarray, dim: int) -> np.ndarray:
        weight = (tau**self.alpha) * (eta**self.beta)  # shape (dim, 2)
        totals = weight.sum(axis=1, keepdims=True)
        probs_select = weight[:, 1:2] / totals  # P(state=1) per feature
        draws = self.rng.random(size=(dim, 1))
        mask = (draws < probs_select).astype(int).ravel()
        if mask.sum() == 0:
            mask[self.rng.integers(0, dim)] = 1
        return mask

    def optimize(self, fitness_fn: FitnessFn, dim: int) -> OptimizationResult:
        tau = np.ones((dim, 2))  # tau[:, 0] = "skip", tau[:, 1] = "select"
        eta = np.ones((dim, 2))  # neutral heuristic (undocumented in source report)

        best_mask = None
        best_fitness = np.inf
        convergence = np.empty(self.max_iter)

        for it in range(self.max_iter):
            masks = np.empty((self.population_size, dim), dtype=int)
            costs = np.empty(self.population_size)

            for k in range(self.population_size):
                mask = self._construct_mask(tau, eta, dim)
                masks[k] = mask
                costs[k] = fitness_fn(mask)
                self.n_evaluations += 1

                if costs[k] < best_fitness:
                    best_fitness = costs[k]
                    best_mask = mask.copy()

            # Global pheromone update: evaporation + deposit from all ants.
            tau *= 1 - self.rho
            deposit = np.zeros_like(tau)
            for k in range(self.population_size):
                contribution = self.Q / (costs[k] + _EPS)
                for j in range(dim):
                    deposit[j, masks[k, j]] += contribution
            tau += deposit
            tau = np.clip(tau, _EPS, None)

            convergence[it] = best_fitness

        return OptimizationResult(
            best_position=best_mask.astype(float),
            best_mask=best_mask,
            best_fitness=float(best_fitness),
            convergence_curve=convergence,
            n_evaluations=self.n_evaluations,
        )
