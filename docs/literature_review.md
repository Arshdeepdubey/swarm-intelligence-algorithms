# Literature Review: Swarm Intelligence for Feature Selection

This is a working synthesis of `docs/source-documents/IEEE_Conference_Paper.docx`
("A survey of swarm intelligence-based feature selection algorithms") and
the literature-survey chapter of `docs/source-documents/SDP_Final_Project_Report.docx`.
It exists so the code in `src/swarm_fs/` can be read against the research
motivation without re-opening the Word documents.

## Problem: feature selection

High-dimensional datasets suffer from the "curse of dimensionality" [Bellman,
1957]: as the number of features `n` grows, the number of possible feature
subsets grows as `2^n`, irrelevant/redundant features add noise and
overfitting risk, and models become harder to train, interpret, and
visualize. Feature selection searches for a small, informative subset of
the original `n` features that preserves (or improves) predictive
performance.

Three families of feature-selection method are surveyed:

| Method | How it scores a subset | Pros | Cons |
|---|---|---|---|
| Filter | Statistical properties of the features alone (e.g. correlation, information gain), independent of any classifier | Cheap, scales to high dimensions, no classifier bias | Ignores feature interactions and the target classifier, generally lower predictive performance |
| Wrapper | Performance of a specific classifier trained on the candidate subset | Best predictive performance; accounts for feature interactions | Computationally expensive — naively exhaustive search over `2^n` subsets is intractable |
| Embedded | Built into the classifier's own training (e.g. LASSO, Random Forest importance) | One training pass; captures feature/model interaction | Tied to one classifier family |

This project's implementation is a **wrapper method**: an SVM classifier's
held-out accuracy is the fitness signal an SI algorithm searches against
(`src/swarm_fs/fitness.py`).

## Why swarm intelligence

Swarm Intelligence (SI) is a population-based branch of Evolutionary
Computation (EC) inspired by decentralized, locally-interacting biological
systems (ant colonies, bee colonies, bird flocks, cuckoo brood parasitism).
Unlike exhaustive or purely greedy search, SI algorithms:

- Make no assumptions about the shape of the search space.
- Maintain multiple candidate solutions in parallel (well suited to a
  `2^n`-sized discrete search space).
- Balance **exploration** (searching new regions) against **exploitation**
  (refining known-good regions), which helps avoid getting trapped in
  local optima — a documented weakness of PSO and ABC in the source paper
  (Sec 5.1, "Algorithm specific issues").

## Binary representation

Because SI algorithms are naturally continuous-valued optimizers but a
feature mask is inherently binary (select / don't select), the source
paper (Sec 3.2.2) describes converting a continuous position `x` to a bit
via the sigmoid transform:

```
sigmoid(x) = 1 / (1 + e^-x)
```

followed by comparing the result to a random threshold: select the
feature if `sigmoid(x)` exceeds the threshold. This exact rule is
implemented in `src/swarm_fs/utils.py::binarize`, and used by the
continuous-representation algorithms (PSO, ABC, Cuckoo Search).
`src/swarm_fs/algorithms/aco.py` instead works natively over `{0, 1}` via
a pheromone-per-feature-state formulation (see that module's docstring for
why ACO doesn't need the sigmoid trick).

## The four algorithms implemented

| Algorithm | Inspiration | Year / authors | Key mechanism |
|---|---|---|---|
| Particle Swarm Optimization (PSO) | Bird flocks / fish schools | Kennedy & Eberhart, 1995 | Particles move toward a blend of their own best-known position and the swarm's best-known position |
| Ant Colony Optimization (ACO) | Ant foraging trails | Dorigo, 1992 | Pheromone trails bias probabilistic construction of solutions; evaporation + deposit reinforce good choices |
| Artificial Bee Colony (ABC) | Honeybee foraging | Karaboga, 2005 | Employed/onlooker/scout bees balance exploitation (neighborhood search) and exploration (random restarts) |
| Cuckoo Search Algorithm (CSA) | Cuckoo brood parasitism | Yang & Deb, 2009 | Levy flights (heavy-tailed random walk) generate new candidate nests; a fraction of the worst nests are abandoned each generation |

The survey paper also discusses several other SI variants applied to
feature selection in the literature — Glowworm Swarm Optimization (GSO),
the Bat Algorithm (BA), and the Firefly Algorithm (FA) — which are
summarized in Sec 4.4 but were **not** part of the SDP report's
implementation scope and are therefore not implemented in this repository
(see `docs/documentation_review.md`).

## Evaluation protocol

Both source documents converge on the same protocol, which
`src/swarm_fs/experiment.py` reproduces:

1. Load one of four `sklearn.datasets` benchmarks: Iris (150 x 4), Breast
   Cancer (569 x 30), Wine (178 x 13), Digits (1797 x 64).
2. Use an SI algorithm to search for a binary feature mask that minimizes
   a weighted combination of `(1 - SVM accuracy)` and `(selected
   features / total features)`.
3. Report **subset accuracy** (SVM trained on the selected features,
   evaluated on a held-out test split) against the **all-features
   accuracy** baseline (SVM trained on every feature).
4. Repeat 10 times per (algorithm, dataset) pair to characterize variance
   across random restarts.

## Key references

Full numbered reference lists are preserved in the source documents. The
most load-bearing citations for this codebase:

- Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *PSO*.
- Karaboga, D. (2005). An idea based on honey bee swarm for numerical
  optimization. Technical Report TR06, Erciyes University.
- Dorigo, M., & Stützle, T. (2004). *Ant colony optimization*. MIT Press.
- Yang, X.-S., & Deb, S. (2009). Cuckoo search via Levy flights. *NaBIC
  2009*.
- Kothari, V. et al. (2012). A survey on particle swarm optimization in
  feature selection. *Global Trends in Information Systems and Software
  Applications*.
