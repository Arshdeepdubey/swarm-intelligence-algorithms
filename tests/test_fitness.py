import numpy as np
import pytest

from swarm_fs.datasets import load_dataset
from swarm_fs.fitness import NO_FEATURE_PENALTY, SVMFeatureSelection


@pytest.fixture(scope="module")
def iris_split():
    return load_dataset("iris", test_size=0.3, random_state=0)


def test_no_features_selected_returns_penalty(iris_split):
    objective = SVMFeatureSelection(iris_split.X_train, iris_split.y_train)
    mask = np.zeros(iris_split.n_features, dtype=int)
    assert objective.evaluate(mask) == pytest.approx(NO_FEATURE_PENALTY)


def test_evaluate_detailed_reports_consistent_feature_count(iris_split):
    objective = SVMFeatureSelection(iris_split.X_train, iris_split.y_train)
    mask = np.array([1, 0, 1, 0])
    result = objective.evaluate_detailed(mask)
    assert result.n_selected == 2
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.fitness <= 1.0


def test_fitness_is_lower_for_higher_accuracy_at_fixed_alpha(iris_split):
    # alpha=1.0 means fitness is purely (1 - accuracy): a full-feature mask
    # (which sklearn's SVM handles well on Iris) should not be worse than
    # a near-random low-accuracy alternative in the same feature-count regime.
    objective = SVMFeatureSelection(iris_split.X_train, iris_split.y_train, alpha=1.0)
    full_mask = np.ones(iris_split.n_features, dtype=int)
    result_full = objective.evaluate_detailed(full_mask)
    assert result_full.fitness == pytest.approx(1 - result_full.accuracy)


def test_alpha_zero_only_penalizes_feature_count(iris_split):
    objective = SVMFeatureSelection(iris_split.X_train, iris_split.y_train, alpha=0.0)
    mask = np.array([1, 1, 0, 0])
    result = objective.evaluate_detailed(mask)
    assert result.fitness == pytest.approx(result.n_selected / iris_split.n_features)


def test_alpha_out_of_range_raises(iris_split):
    with pytest.raises(ValueError):
        SVMFeatureSelection(iris_split.X_train, iris_split.y_train, alpha=1.5)


def test_accuracy_on_matches_manual_sklearn_fit(iris_split):
    from sklearn.svm import SVC

    objective = SVMFeatureSelection(iris_split.X_train, iris_split.y_train)
    mask = np.array([1, 1, 1, 1], dtype=bool)

    expected_clf = SVC(kernel="linear", random_state=0)
    expected_clf.fit(iris_split.X_train[:, mask], iris_split.y_train)
    expected_acc = expected_clf.score(iris_split.X_test[:, mask], iris_split.y_test)

    got_acc = objective.accuracy_on(mask.astype(int), iris_split.X_test, iris_split.y_test)
    assert got_acc == pytest.approx(expected_acc)


def test_all_features_accuracy_is_between_zero_and_one(iris_split):
    objective = SVMFeatureSelection(iris_split.X_train, iris_split.y_train)
    acc = objective.all_features_accuracy(iris_split.X_test, iris_split.y_test)
    assert 0.0 <= acc <= 1.0
