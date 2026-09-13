from __future__ import annotations

import numpy as np
import pytest

import mytorch as mt

pytestmark = pytest.mark.gpu


def _labels(loader: mt.data.DataLoader) -> list[int]:
    result = []
    for _, labels in loader:
        result.extend(int(value) for value in labels.numpy())
    return result


def test_tensor_dataset_maps_samples_and_vectorized_batches() -> None:
    features = mt.arange(20).reshape(10, 2)
    labels = mt.arange(10, dtype=mt.int64)
    dataset = mt.data.TensorDataset(features, labels)

    sample_features, sample_label = dataset[-1]
    np.testing.assert_array_equal(sample_features.numpy(), [18, 19])
    assert sample_label.item() == 9

    batches = list(mt.data.DataLoader(dataset, batch_size=4))
    assert len(batches) == 3
    assert batches[0][0].shape == (4, 2)
    assert batches[0][1].shape == (4,)
    assert batches[-1][0].shape == (2, 2)
    assert all(value.device == "cuda:0" for batch in batches for value in batch)


def test_loader_shuffle_is_seeded_and_changes_between_epochs() -> None:
    dataset = mt.data.TensorDataset(
        mt.arange(12).unsqueeze(1), mt.arange(12, dtype=mt.int64)
    )
    first = mt.data.DataLoader(dataset, batch_size=3, shuffle=True, seed=42)
    second = mt.data.DataLoader(dataset, batch_size=3, shuffle=True, seed=42)

    first_epoch = _labels(first)
    second_epoch = _labels(first)
    assert first_epoch == _labels(second)
    assert second_epoch == _labels(second)
    assert first_epoch != second_epoch
    assert sorted(first_epoch) == list(range(12))


def test_drop_last_and_loader_length() -> None:
    dataset = mt.data.TensorDataset(mt.arange(10))
    keep = mt.data.DataLoader(dataset, batch_size=4)
    drop = mt.data.DataLoader(dataset, batch_size=4, drop_last=True)

    assert len(keep) == 3
    assert len(drop) == 2
    assert [batch[0].shape[0] for batch in keep] == [4, 4, 2]
    assert [batch[0].shape[0] for batch in drop] == [4, 4]


class StructuredDataset(mt.data.Dataset[dict[str, object]]):
    def __len__(self) -> int:
        return 5

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "input": np.asarray([index, index + 1], dtype=np.float32),
            "target": index % 2,
            "metadata": f"sample-{index}",
        }


def test_default_collate_recursively_batches_structured_samples() -> None:
    batch = next(iter(mt.data.DataLoader(StructuredDataset(), batch_size=3)))

    assert set(batch) == {"input", "target", "metadata"}
    assert batch["input"].shape == (3, 2)
    assert batch["input"].dtype == mt.float32
    assert batch["target"].dtype == mt.int64
    assert batch["metadata"] == ["sample-0", "sample-1", "sample-2"]


def test_custom_collate_receives_sample_list() -> None:
    dataset = mt.data.TensorDataset(mt.arange(6))
    loader = mt.data.DataLoader(
        dataset,
        batch_size=3,
        collate_fn=lambda samples: mt.stack([sample[0] for sample in samples]) + 10,
    )

    np.testing.assert_array_equal(next(iter(loader)).numpy(), [10, 11, 12])


def test_subset_and_random_split_are_disjoint_and_reproducible() -> None:
    dataset = mt.data.TensorDataset(
        mt.arange(10).unsqueeze(1), mt.arange(10, dtype=mt.int64)
    )
    train, valid = mt.data.random_split(dataset, [7, 3], seed=9)
    repeated_train, repeated_valid = mt.data.random_split(dataset, [7, 3], seed=9)

    assert train.indices == repeated_train.indices
    assert valid.indices == repeated_valid.indices
    assert set(train.indices).isdisjoint(valid.indices)
    assert set(train.indices) | set(valid.indices) == set(range(10))
    assert len(list(mt.data.DataLoader(train, batch_size=4))) == 2


def test_data_validation_errors_are_clear() -> None:
    with pytest.raises(ValueError, match="at least one"):
        mt.data.TensorDataset()
    with pytest.raises(ValueError, match="equal lengths"):
        mt.data.TensorDataset(mt.ones(2, 3), mt.ones(3))
    with pytest.raises(IndexError, match="out of range"):
        mt.data.TensorDataset(mt.ones(2))[2]
    with pytest.raises(ValueError, match="positive integer"):
        mt.data.DataLoader(mt.data.TensorDataset(mt.ones(2)), batch_size=0)
    with pytest.raises(ValueError, match="sum"):
        mt.data.random_split(mt.data.TensorDataset(mt.ones(3)), [1, 1], seed=1)
    with pytest.raises(ValueError, match="non-empty"):
        mt.data.default_collate([])
