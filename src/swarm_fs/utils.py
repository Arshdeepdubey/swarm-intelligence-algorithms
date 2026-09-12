"""
Shared numerical helpers used by every swarm-intelligence optimizer.

The IEEE survey paper (Section 3.2.2, "Binary Representation") and the SDP
report describe the same continuous-to-binary conversion scheme used across
PSO/ABC/CSA:

    1. Squash a real-valued position through the sigmoid function.
    2. Compare the squashed value against a uniform random threshold.
    3. Emit 1 (feature selected) if the squashed value exceeds the
       threshold, else 0.

This module centralizes that logic (and a couple of small utilities) so
every algorithm module shares one, tested implementation instead of
re-deriving it.
"""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid: f(x) = 1 / (1 + e^-x).

    Uses the standard split-branch formulation so large-magnitude inputs
    (positive or negative) never overflow ``np.exp``.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def binarize(position: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Convert a continuous position vector into a binary feature mask.

    Matches the paper's rule: sigmoid the continuous value, draw a random
    threshold in [0, 1) per dimension, and select the feature (1) when the
    sigmoid output exceeds the threshold.
    """
    probs = sigmoid(position)
    thresholds = rng.random(size=probs.shape)
    return (probs > thresholds).astype(int)


def ensure_at_least_one_feature(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Guarantee a binary mask selects >=1 feature.

    An all-zero mask is meaningless for a classifier (no input features),
    so if binarization collapses to all zeros we flip a single,
    randomly-chosen bit on. This mirrors the "if no features are selected,
    return default value" safeguard described for the fitness function in
    the SDP report, but fixes it at the representation level so downstream
    algorithm logic never has to special-case an empty mask.
    """
    mask = mask.copy()
    if mask.sum() == 0:
        idx = rng.integers(0, mask.shape[0])
        mask[idx] = 1
    return mask


def clip(position: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Clip a position vector to the search-space bounds (Table 7: [0, 1])."""
    return np.clip(position, lower, upper)


def levy_flight(dim: int, rng: np.random.Generator, beta: float = 1.5) -> np.ndarray:
    """Draw a step from a Mantegna-approximated Levy-stable distribution.

    Used by the Cuckoo Search Algorithm, per the IEEE paper's reference to
    "Levy flights and random walks" (Sec 4.3 / SDP Sec 3.3 CSA equations).
    """
    num = math_gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = math_gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma_u = (num / den) ** (1 / beta)

    u = rng.normal(0, sigma_u, size=dim)
    v = rng.normal(0, 1, size=dim)
    step = u / (np.abs(v) ** (1 / beta))
    return step


def math_gamma(x: float) -> float:
    """Thin wrapper so this module has no import-time dependency on scipy."""
    from math import gamma

    return gamma(x)
