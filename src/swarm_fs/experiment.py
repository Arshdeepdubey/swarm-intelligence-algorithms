"""
Experiment runner: reproduces the "run each algorithm N times per dataset,
compare subset accuracy vs. all-features accuracy" protocol described in
the SDP report (Sec 4.3, "Experimental Outcomes"):

    "This is the ABC algorithm's output when ran 10 times against each of
    the Iris, Breast Cancer, Wine and Digits datasets. Subset Acc denotes
    the accuracy of the chosen features on each run and No. of features
    shows the amount of features being selected over each single
    iteration."
    (repeated verbatim in the report for CSA, ACO, and PSO)

To avoid feature-selection leakage into the reported test accuracy, the
optimizer's internal fitness function only ever sees a train/validation
split carved out of the training data; the dataset's held-out test split
is used exclusively for the final "subset accuracy" / "all features
accuracy" numbers below.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from .algorithms import ALGORITHMS
from .datasets import DatasetSplit, available_datasets, load_dataset
from .fitness import SVMFeatureSelection

DEFAULT_ALGORITHMS = tuple(ALGORITHMS.keys())
DEFAULT_DATASETS = tuple(available_datasets())


def run_experiment(
    algorithm_name: str,
    dataset_name: str,
    n_runs: int = 10,
    population_size: int = 10,
    max_iter: int = 100,
    alpha: float = 0.99,
    kernel: str = "linear",
    base_seed: int = 0,
    dataset: DatasetSplit | None = None,
) -> pd.DataFrame:
    """Run one SI algorithm ``n_runs`` times on one dataset.

    Returns a tidy DataFrame with one row per run.
    """
    if algorithm_name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algorithm_name!r}. Available: {list(ALGORITHMS)}")

    data = dataset if dataset is not None else load_dataset(dataset_name)
    algo_cls = ALGORITHMS[algorithm_name]

    records = []
    for run in range(n_runs):
        seed = base_seed + run

        # Inner split used only to *search* for a good feature mask.
        X_fit_train, X_fit_val, y_fit_train, y_fit_val = train_test_split(
            data.X_train, data.y_train, test_size=0.25, random_state=seed, stratify=data.y_train
        )
        objective = SVMFeatureSelection(
            X_fit_train, y_fit_train, X_fit_val, y_fit_val, alpha=alpha, kernel=kernel
        )

        optimizer = algo_cls(population_size=population_size, max_iter=max_iter, seed=seed)
        result = optimizer.optimize(objective.evaluate, dim=data.n_features)

        # Final reporting fitness object: trained on the *full* training
        # split, scored against the held-out test split.
        reporting = SVMFeatureSelection(data.X_train, data.y_train, kernel=kernel)
        subset_accuracy = reporting.accuracy_on(result.best_mask, data.X_test, data.y_test)
        all_features_accuracy = reporting.all_features_accuracy(data.X_test, data.y_test)

        records.append(
            {
                "algorithm": algorithm_name,
                "dataset": dataset_name,
                "run": run,
                "seed": seed,
                "n_selected": int(result.best_mask.sum()),
                "n_total_features": data.n_features,
                "subset_accuracy": subset_accuracy,
                "all_features_accuracy": all_features_accuracy,
                "internal_fitness": result.best_fitness,
                "n_evaluations": result.n_evaluations,
            }
        )

    return pd.DataFrame.from_records(records)


def run_all_experiments(
    algorithms: Iterable[str] = DEFAULT_ALGORITHMS,
    datasets: Iterable[str] = DEFAULT_DATASETS,
    n_runs: int = 10,
    population_size: int = 10,
    max_iter: int = 100,
    alpha: float = 0.99,
    kernel: str = "linear",
    base_seed: int = 0,
    progress: bool = False,
) -> pd.DataFrame:
    """Run every (algorithm, dataset) combination and concatenate results."""
    frames = []
    for dataset_name in datasets:
        data = load_dataset(dataset_name)
        for algorithm_name in algorithms:
            if progress:
                print(f"[run_all_experiments] {algorithm_name} on {dataset_name} ...")
            frames.append(
                run_experiment(
                    algorithm_name,
                    dataset_name,
                    n_runs=n_runs,
                    population_size=population_size,
                    max_iter=max_iter,
                    alpha=alpha,
                    kernel=kernel,
                    base_seed=base_seed,
                    dataset=data,
                )
            )
    return pd.concat(frames, ignore_index=True)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-run results into mean/std summary rows per (algorithm, dataset)."""
    summary = (
        df.groupby(["dataset", "algorithm"])
        .agg(
            mean_subset_accuracy=("subset_accuracy", "mean"),
            std_subset_accuracy=("subset_accuracy", "std"),
            mean_all_features_accuracy=("all_features_accuracy", "mean"),
            mean_n_selected=("n_selected", "mean"),
            n_total_features=("n_total_features", "first"),
            n_runs=("run", "count"),
        )
        .reset_index()
    )
    summary["accuracy_gain"] = (
        summary["mean_subset_accuracy"] - summary["mean_all_features_accuracy"]
    )
    return summary.sort_values(["dataset", "mean_subset_accuracy"], ascending=[True, False])
