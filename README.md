# Swarm Intelligence Algorithms for Optimal Feature Selection

A study, implementation, and reproducible experimental pipeline for four
Swarm Intelligence (SI) metaheuristics — **Particle Swarm Optimization
(PSO)**, **Ant Colony Optimization (ACO)**, **Artificial Bee Colony
Optimization (ABC)**, and the **Cuckoo Search Algorithm (CSA)** — applied
to feature selection for Support Vector Machine (SVM) classifiers.

This repository turns the team's IEEE survey paper and Senior Design
Project (SDP) report into working, tested code: every algorithm and
dataset described in the source documents (see `docs/source-documents/`)
is implemented in `src/swarm_fs/`, exercised by 72 automated tests
(`tests/`), and backed by a real 160-run experimental sweep
(`results/demo_results.csv`).

**Team:** Abhipsha Sahu, Sanjivani Sharan, Arshdeep Dubey, Abhisman Sarkar
**Supervisor:** Dr. Subrat Kumar Nayak
**Institution:** Dept. of CSE, Faculty of Engineering & Technology (ITER),
Siksha 'O' Anusandhan (Deemed to be) University, Bhubaneswar, Odisha

## What this project does

For each of four benchmark datasets (Iris, Breast Cancer, Wine, Digits —
all from `sklearn.datasets`), each SI algorithm searches for a binary
feature mask that maximizes SVM classification accuracy while minimizing
the number of features used. The result is compared against a baseline
SVM trained on every feature, across 10 independent runs per
(algorithm, dataset) pair — see `results/README.md` for the recorded
outcome of that sweep.

## Project structure

```
swarm-intelligence-algorithms/
├── src/swarm_fs/                  # The package: importable, tested code
│   ├── algorithms/
│   │   ├── base.py                # Shared bounds/seeding/binarization plumbing
│   │   ├── pso.py                 # Particle Swarm Optimization
│   │   ├── aco.py                 # (Binary/pheromone) Ant Colony Optimization
│   │   ├── abc.py                 # Artificial Bee Colony Optimization
│   │   └── cuckoo.py              # Cuckoo Search Algorithm (Levy flights)
│   ├── benchmarks.py               # Sphere/Step/Ackley/Penalized test functions
│   ├── datasets.py                 # Iris/Breast Cancer/Wine/Digits loading + splitting
│   ├── fitness.py                  # SVM-wrapper fitness function (SVMFeatureSelection)
│   ├── experiment.py               # N-run experiment driver + summary aggregation
│   └── utils.py                    # sigmoid, binarize, Levy-flight sampling
├── scripts/
│   ├── run_experiments.py          # CLI: reproduce the SDP report's experiment sweep
│   └── compare_algorithms.py       # CLI: summary table + comparison chart
├── tests/                          # 72 pytest cases (unit + integration + smoke)
├── docs/
│   ├── source-documents/           # The original IEEE paper, SDP report, and PPT
│   ├── literature_review.md        # Synthesis of the survey paper's background
│   └── documentation_review.md     # Cross-check: paper vs. report vs. PPT vs. code
├── results/
│   ├── demo_results.csv            # 160 recorded runs (4 algos x 4 datasets x 10 runs)
│   ├── summary.csv                 # Aggregated mean/std per (dataset, algorithm)
│   ├── comparison.png              # Bar chart: subset accuracy vs. all-features baseline
│   └── README.md                   # How this was generated + what it shows
├── pyproject.toml / requirements.txt
└── LICENSE
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the test suite (fast: ~3-4 seconds, 72 tests)
pytest -q

# Reproduce a quick smoke-scale experiment
python scripts/run_experiments.py --datasets iris --algorithms pso --n-runs 2 \
    --output results/quick_test.csv

# Reproduce the full SDP-report-scale sweep (population=10, iterations=100,
# 10 runs, all 4 algorithms x all 4 datasets) -- takes roughly 12-15 minutes,
# dominated by the Digits dataset's ABC runs
python scripts/run_experiments.py --output results/demo_results.csv
python scripts/compare_algorithms.py --input results/demo_results.csv --output-dir results
```

## Using the library directly

```python
from swarm_fs.datasets import load_dataset
from swarm_fs.fitness import SVMFeatureSelection
from swarm_fs.algorithms import ALGORITHMS

data = load_dataset("wine")
objective = SVMFeatureSelection(data.X_train, data.y_train)

pso = ALGORITHMS["pso"](population_size=10, max_iter=100, seed=0)
result = pso.optimize(objective.evaluate, dim=data.n_features)

print(result.best_mask, result.best_fitness)
```

## Documentation

- `docs/literature_review.md` — background on feature selection and each
  SI algorithm, synthesized from the IEEE survey paper.
- `docs/documentation_review.md` — a structured comparison of the IEEE
  paper, the SDP report, the preliminary evaluation slide deck, and this
  codebase: what's consistent, what's missing from the source documents
  (e.g. undocumented hyperparameters), and what the actual experimental
  results show versus the report's stated conclusion.
- `results/README.md` — how the recorded experimental sweep was produced
  and how to reproduce it.

## License

MIT — see `LICENSE`.
