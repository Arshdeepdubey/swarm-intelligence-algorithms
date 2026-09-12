"""
Swarm-intelligence optimizers.

Four algorithms are implemented here, matching the equations quoted in the
IEEE survey paper (Sec 4) and the SDP report (Sec 3.3):

* :mod:`swarm_fs.algorithms.pso`    -- Particle Swarm Optimization
* :mod:`swarm_fs.algorithms.aco`    -- (Binary) Ant Colony Optimization
* :mod:`swarm_fs.algorithms.abc`    -- Artificial Bee Colony Optimization
* :mod:`swarm_fs.algorithms.cuckoo` -- Cuckoo Search Algorithm

Every optimizer shares the same public contract: ``optimize(fitness_fn,
dim, ...) -> OptimizationResult``, where ``fitness_fn`` maps a binary
NumPy array of length ``dim`` to a scalar cost (lower is better). This
lets :mod:`swarm_fs.experiment` treat all four interchangeably.
"""

from .abc import ArtificialBeeColony
from .aco import AntColonyOptimization
from .base import OptimizationResult
from .cuckoo import CuckooSearch
from .pso import ParticleSwarmOptimization

ALGORITHMS = {
    "pso": ParticleSwarmOptimization,
    "aco": AntColonyOptimization,
    "abc": ArtificialBeeColony,
    "cuckoo": CuckooSearch,
}

__all__ = [
    "OptimizationResult",
    "ParticleSwarmOptimization",
    "AntColonyOptimization",
    "ArtificialBeeColony",
    "CuckooSearch",
    "ALGORITHMS",
]
