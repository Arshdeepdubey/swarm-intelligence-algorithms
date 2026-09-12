"""
SVM-wrapped fitness function for feature selection.

Directly implements the ``SVMFeatureSelection`` class described in the SDP
report (Sec 3.3, "Feature Selection using SI algorithms"):

    Initialize with training data and alpha parameter.
    Define evaluation function:
        Select features based on binary values.
        Calculate number of selected features.
        If no features are selected, return default (worst) value.
        Otherwise, train an SVM classifier on the selected features and
        calculate accuracy.
        Calculate score based on accuracy and feature ratio.
        Return weighted score.

All four algorithms in this package are formulated as *minimizers*, so the
fitness returned here is a cost: lower is better. Following the standard
wrapper-based feature-selection formulation used across the SI-FS
literature the paper surveys (e.g. Kothari et al. 2012, ref [13]):

    fitness = alpha * (1 - accuracy) + (1 - alpha) * (k / n)

where ``k`` is the number of selected features and ``n`` is the total
number of features. ``alpha`` close to 1 (the report's SDP code discussion
implies a high weight on accuracy) trades a small amount of dimensionality
reduction credit for classification performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.svm import SVC

# Worst-case fitness returned when a candidate solution selects zero
# features (a degenerate, unusable classifier). Matches the SDP report's
# "if no features are selected, return default value" safeguard.
NO_FEATURE_PENALTY = 1.0


@dataclass
class EvaluationResult:
    fitness: float
    accuracy: float
    n_selected: int


class SVMFeatureSelection:
    """Wrapper-method fitness function: SVM accuracy vs. feature count."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        alpha: float = 0.99,
        kernel: str = "linear",
        C: float = 1.0,
        random_state: int | None = 0,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")

        self.X_train = X_train
        self.y_train = y_train
        # A held-out validation split is used *inside* the fitness function
        # so the optimizer never sees the final test set -- avoiding
        # feature-selection leakage into the reported test accuracy.
        self.X_eval = X_test if X_test is not None else X_train
        self.y_eval = y_test if y_test is not None else y_train
        self.alpha = alpha
        self.kernel = kernel
        self.C = C
        self.random_state = random_state
        self.n_features = X_train.shape[1]

    def _select(self, binary_mask: np.ndarray):
        mask = np.asarray(binary_mask).astype(bool)
        return self.X_train[:, mask], self.X_eval[:, mask]

    def evaluate(self, binary_mask: np.ndarray) -> float:
        """Return the scalar fitness (lower is better) for a feature mask."""
        return self.evaluate_detailed(binary_mask).fitness

    def evaluate_detailed(self, binary_mask: np.ndarray) -> EvaluationResult:
        binary_mask = np.asarray(binary_mask).astype(int)
        n_selected = int(binary_mask.sum())

        if n_selected == 0:
            return EvaluationResult(
                fitness=NO_FEATURE_PENALTY, accuracy=0.0, n_selected=0
            )

        X_train_sub, X_eval_sub = self._select(binary_mask)
        clf = SVC(kernel=self.kernel, C=self.C, random_state=self.random_state)
        clf.fit(X_train_sub, self.y_train)
        accuracy = clf.score(X_eval_sub, self.y_eval)

        feature_ratio = n_selected / self.n_features
        fitness = self.alpha * (1 - accuracy) + (1 - self.alpha) * feature_ratio
        return EvaluationResult(
            fitness=fitness, accuracy=accuracy, n_selected=n_selected
        )

    def accuracy_on(
        self, binary_mask: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
    ) -> float:
        """Train on the selected subset and score against an external test set.

        Used after optimization finishes to report the final "subset
        accuracy" the SDP report tracks per run.
        """
        mask = np.asarray(binary_mask).astype(bool)
        if mask.sum() == 0:
            return 0.0
        clf = SVC(kernel=self.kernel, C=self.C, random_state=self.random_state)
        clf.fit(self.X_train[:, mask], self.y_train)
        return clf.score(X_test[:, mask], y_test)

    def all_features_accuracy(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> float:
        """Baseline accuracy using every feature (no selection)."""
        clf = SVC(kernel=self.kernel, C=self.C, random_state=self.random_state)
        clf.fit(self.X_train, self.y_train)
        return clf.score(X_test, y_test)
