"""
Classic continuous optimization benchmark functions.

The SDP report (Sec 3, Introduction) states: "Different SI algorithms are
also compared using different Objective functions such as sphere function,
step function, Ackley function and penalized function." Those four
functions are implemented here so the optimizers can be validated against
known global optima independently of the (stochastic, dataset-dependent)
SVM fitness -- this is what the unit tests in ``tests/test_algorithms.py``
exercise to confirm each optimizer actually converges.

All four functions are defined as minimization problems with a global
minimum of 0 at (or near) the origin, following their standard textbook
definitions.
"""

from __future__ import annotations

import numpy as np


def sphere(x: np.ndarray) -> float:
    """f(x) = sum(x_i^2). Global minimum: f(0) = 0."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2))


def step(x: np.ndarray) -> float:
    """f(x) = sum((floor(x_i + 0.5))^2). Global minimum: f(0) = 0."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(np.floor(x + 0.5) ** 2))


def ackley(x: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi) -> float:
    """Ackley function. Global minimum: f(0) = 0."""
    x = np.asarray(x, dtype=float)
    d = x.size
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / d))
    term2 = -np.exp(sum_cos / d)
    return float(term1 + term2 + a + np.e)


def _u(x: np.ndarray, a: float, k: float, m: float) -> np.ndarray:
    return np.where(
        x > a,
        k * (x - a) ** m,
        np.where(x < -a, k * (-x - a) ** m, 0.0),
    )


def penalized(x: np.ndarray) -> float:
    """Generalized penalized function #1. Global minimum: f(-1,...,-1) = 0.

    Standard definition used in swarm-intelligence benchmarking suites
    (e.g. Yao, Liu & Lin, 1999).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    y = 1 + (x + 1) / 4

    term1 = 10 * (np.sin(np.pi * y[0]) ** 2)
    term2 = np.sum((y[:-1] - 1) ** 2 * (1 + 10 * (np.sin(np.pi * y[1:]) ** 2)))
    term3 = (y[-1] - 1) ** 2
    penalty = np.sum(_u(x, 10, 100, 4))

    return float((np.pi / n) * (term1 + term2 + term3) + penalty)


BENCHMARKS = {
    "sphere": {"fn": sphere, "bounds": (-100.0, 100.0), "optimum": 0.0},
    "step": {"fn": step, "bounds": (-100.0, 100.0), "optimum": 0.0},
    "ackley": {"fn": ackley, "bounds": (-32.0, 32.0), "optimum": 0.0},
    "penalized": {"fn": penalized, "bounds": (-50.0, 50.0), "optimum": 0.0},
}
