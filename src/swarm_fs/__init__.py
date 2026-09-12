"""
swarm_fs
========

A study on Swarm Intelligence Algorithms for Optimal Feature Selection.

This package implements the four swarm-intelligence (SI) metaheuristics
described in the accompanying IEEE survey paper and Senior Design Project
(SDP) report:

* Particle Swarm Optimization (PSO)
* Ant Colony Optimization (ACO)         -- binary/pheromone variant
* Artificial Bee Colony Optimization (ABC)
* Cuckoo Search Algorithm (CSA)

Each algorithm searches for a binary feature mask that maximizes the
accuracy of a Support Vector Machine (SVM) classifier while minimizing the
number of selected features, on four benchmark datasets from
``sklearn.datasets``: Iris, Breast Cancer, Wine, and Digits.
"""

__version__ = "1.0.0"

from .fitness import SVMFeatureSelection
from .experiment import run_experiment, run_all_experiments

__all__ = [
    "SVMFeatureSelection",
    "run_experiment",
    "run_all_experiments",
]
