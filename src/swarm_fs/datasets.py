"""
Dataset loading utilities.

The SDP report (Sec 3.1, Table 1) uses four datasets pulled from
``sklearn.datasets``: Iris, Breast Cancer, Wine, and Digits. This module
centralizes their loading, feature-name extraction, and a stratified
train/test split so the experiment runner and the tests can share one
code path (previously the report describes this being done ad hoc inside
each algorithm's driver script).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET_LOADERS = {
    "iris": load_iris,
    "breast_cancer": load_breast_cancer,
    "wine": load_wine,
    "digits": load_digits,
}

# Matches Table 1 in the SDP report (Dataset(s) Description).
DATASET_METADATA = {
    "iris": {"instances": 150, "features": 4, "classes": 3},
    "breast_cancer": {"instances": 569, "features": 30, "classes": 2},
    "wine": {"instances": 178, "features": 13, "classes": 3},
    "digits": {"instances": 1797, "features": 64, "classes": 10},
}


@dataclass
class DatasetSplit:
    """Container for a train/test split plus the original feature names."""

    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: Sequence[str]

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


def available_datasets() -> list:
    """Names of the datasets this package knows how to load."""
    return sorted(DATASET_LOADERS.keys())


def load_dataset(
    name: str,
    test_size: float = 0.3,
    random_state: int | None = 42,
    scale: bool = True,
) -> DatasetSplit:
    """Load one of the four SDP-report datasets and split it.

    Parameters
    ----------
    name:
        One of ``iris``, ``breast_cancer``, ``wine``, ``digits``.
    test_size:
        Fraction held out for testing (SVM evaluation), default 30%.
    random_state:
        Seed for the split, for reproducibility.
    scale:
        Whether to standardize features (zero mean, unit variance) before
        splitting. SVMs are scale-sensitive, and the report's use of
        heterogeneous-unit datasets (e.g. Wine's chemical concentrations)
        makes this important for a fair comparison across algorithms.
    """
    if name not in DATASET_LOADERS:
        raise ValueError(
            f"Unknown dataset {name!r}. Available: {available_datasets()}"
        )

    bunch = DATASET_LOADERS[name]()
    X, y = bunch.data, bunch.target
    feature_names = list(getattr(bunch, "feature_names", range(X.shape[1])))
    feature_names = [str(f) for f in feature_names]

    if scale:
        X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return DatasetSplit(
        name=name,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
    )
