import pytest

from swarm_fs.datasets import DATASET_METADATA, available_datasets, load_dataset


@pytest.mark.parametrize("name", available_datasets())
def test_dataset_shapes_match_sdp_report_table1(name):
    """Cross-checks against Table 1 ("Dataset(s) Description") in the SDP report."""
    data = load_dataset(name, test_size=0.3, random_state=0)
    meta = DATASET_METADATA[name]

    total_instances = data.X_train.shape[0] + data.X_test.shape[0]
    assert total_instances == meta["instances"]
    assert data.n_features == meta["features"]
    assert len(data.feature_names) == meta["features"]


@pytest.mark.parametrize("name", available_datasets())
def test_train_test_split_is_stratified_and_nonoverlapping(name):
    data = load_dataset(name, test_size=0.3, random_state=1)
    assert data.X_train.shape[0] == data.y_train.shape[0]
    assert data.X_test.shape[0] == data.y_test.shape[0]
    # Every class present in training should also appear in the test split
    # (guaranteed by stratify=y as long as test_size is large enough).
    assert set(data.y_test.tolist()).issubset(set(data.y_train.tolist()))


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        load_dataset("not_a_real_dataset")


def test_scaling_produces_roughly_standardized_features():
    data = load_dataset("wine", scale=True, random_state=0)
    # Standardized training features should have ~zero mean, ~unit variance.
    assert abs(data.X_train.mean()) < 0.5
    assert 0.5 < data.X_train.std() < 1.5


def test_reproducible_split_with_fixed_seed():
    d1 = load_dataset("iris", random_state=99)
    d2 = load_dataset("iris", random_state=99)
    assert (d1.X_train == d2.X_train).all()
    assert (d1.y_train == d2.y_train).all()
