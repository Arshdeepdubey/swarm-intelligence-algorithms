"""
End-to-end smoke tests for the experiment runner. Kept intentionally tiny
(small population/iteration counts, single run, the fast Iris dataset) so
the full test suite stays fast; ``scripts/run_experiments.py`` is where
the full SDP-report-scale protocol (population=10, iterations=100,
n_runs=10, all 4 datasets) actually gets exercised (see
``results/demo_results.csv`` for a recorded run).
"""

from __future__ import annotations

import pandas as pd
import pytest

from swarm_fs.algorithms import ALGORITHMS
from swarm_fs.datasets import load_dataset
from swarm_fs.experiment import run_all_experiments, run_experiment, summarize_results


@pytest.mark.parametrize("algorithm_name", list(ALGORITHMS.keys()))
def test_run_experiment_returns_expected_columns(algorithm_name):
    df = run_experiment(
        algorithm_name,
        "iris",
        n_runs=2,
        population_size=5,
        max_iter=5,
    )
    assert len(df) == 2
    expected_cols = {
        "algorithm",
        "dataset",
        "run",
        "seed",
        "n_selected",
        "n_total_features",
        "subset_accuracy",
        "all_features_accuracy",
        "internal_fitness",
        "n_evaluations",
    }
    assert expected_cols.issubset(df.columns)
    assert (df["algorithm"] == algorithm_name).all()
    assert (df["dataset"] == "iris").all()
    assert df["n_total_features"].iloc[0] == 4
    assert (df["n_selected"] >= 1).all()
    assert df["subset_accuracy"].between(0, 1).all()
    assert df["all_features_accuracy"].between(0, 1).all()


def test_run_experiment_unknown_algorithm_raises():
    with pytest.raises(ValueError):
        run_experiment("not_an_algorithm", "iris", n_runs=1)


def test_run_experiment_reuses_provided_dataset_split():
    data = load_dataset("wine", random_state=5)
    df = run_experiment("pso", "wine", n_runs=1, population_size=5, max_iter=5, dataset=data)
    assert df["n_total_features"].iloc[0] == data.n_features


def test_run_all_experiments_covers_every_combination():
    df = run_all_experiments(
        algorithms=["pso", "aco"],
        datasets=["iris", "wine"],
        n_runs=1,
        population_size=5,
        max_iter=5,
    )
    assert len(df) == 2 * 2 * 1
    combos = set(zip(df["algorithm"], df["dataset"]))
    assert combos == {("pso", "iris"), ("pso", "wine"), ("aco", "iris"), ("aco", "wine")}


def test_summarize_results_shape():
    df = run_all_experiments(
        algorithms=["pso", "abc"],
        datasets=["iris"],
        n_runs=2,
        population_size=5,
        max_iter=5,
    )
    summary = summarize_results(df)
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 2  # one row per algorithm, single dataset
    assert {"mean_subset_accuracy", "accuracy_gain", "n_runs"}.issubset(summary.columns)
    assert (summary["n_runs"] == 2).all()
