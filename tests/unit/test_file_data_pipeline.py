from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import mytorch as mt


def _image_tree(root: Path) -> None:
    for class_name, color in [("blue", (0, 0, 255)), ("red", (255, 0, 0))]:
        directory = root / class_name
        directory.mkdir(parents=True)
        for index in range(3):
            array = np.full((10, 12, 3), color, dtype=np.uint8)
            array[:, index : index + 2] = 255
            Image.fromarray(array).save(directory / f"{index}.png")


def test_image_folder_transforms_and_threaded_gpu_batches(tmp_path) -> None:
    root = tmp_path / "images"
    _image_tree(root)
    transform = mt.data.transforms.Compose(
        [
            mt.data.transforms.Resize((8, 8)),
            mt.data.transforms.RandomHorizontalFlip(0.5),
            mt.data.transforms.ToArray(),
            mt.data.transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    dataset = mt.data.ImageFolder(root, transform=transform)
    assert dataset.classes == ["blue", "red"]
    assert dataset.class_to_idx == {"blue": 0, "red": 1}
    assert len(dataset) == 6

    synchronous = mt.data.DataLoader(
        dataset, batch_size=2, shuffle=True, seed=11, pin_memory=False
    )
    threaded = mt.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        seed=11,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
    )
    sync_batches = [(x.numpy(), y.numpy()) for x, y in synchronous]
    thread_batches = [(x.numpy(), y.numpy()) for x, y in threaded]
    for expected, actual in zip(sync_batches, thread_batches, strict=True):
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
    assert thread_batches[0][0].shape == (2, 3, 8, 8)
    threaded.close()


def test_csv_and_npy_datasets(tmp_path) -> None:
    csv_path = tmp_path / "samples.csv"
    csv_path.write_text("x,y,label\n1,2,0\n3,4,1\n", encoding="utf-8")
    csv_data = mt.data.CSVDataset(csv_path, target_column="label")
    features, target = csv_data[1]
    np.testing.assert_array_equal(features, [3, 4])
    assert features.dtype == np.float32 and target == 1

    data_path = tmp_path / "data.npy"
    target_path = tmp_path / "target.npy"
    np.save(data_path, np.arange(12, dtype=np.float32).reshape(4, 3))
    np.save(target_path, np.arange(4, dtype=np.int64))
    npy_data = mt.data.NpyDataset(data_path, target_path)
    batch, labels = next(iter(mt.data.DataLoader(npy_data, batch_size=3)))
    np.testing.assert_array_equal(batch.numpy(), np.arange(9).reshape(3, 3))
    np.testing.assert_array_equal(labels.numpy(), [0, 1, 2])


def test_samplers_and_iterable_dataset() -> None:
    source = list(range(5))
    first = list(mt.data.RandomSampler(source, seed=4))
    second = list(mt.data.RandomSampler(source, seed=4))
    assert first == second and sorted(first) == source
    assert (
        len(list(mt.data.BatchSampler(mt.data.SequentialSampler(source), 2, False)))
        == 3
    )

    weighted = mt.data.WeightedRandomSampler([0, 0, 1], 5, seed=2)
    assert list(weighted) == [2] * 5

    class Stream(mt.data.IterableDataset[int]):
        def __iter__(self):
            yield from range(5)

    batches = [
        batch.numpy().tolist() for batch in mt.data.DataLoader(Stream(), batch_size=2)
    ]
    assert batches == [[0, 1], [2, 3], [4]]
    with pytest.raises(ValueError, match="does not support"):
        mt.data.DataLoader(Stream(), shuffle=True)


def test_worker_error_reports_batch_indices() -> None:
    class Broken(mt.data.Dataset[np.ndarray]):
        def __len__(self):
            return 3

        def __getitem__(self, index):
            if index == 1:
                raise OSError("decode failed")
            return np.asarray([index], dtype=np.float32)

    loader = mt.data.DataLoader(Broken(), batch_size=2, num_workers=2)
    with pytest.raises(RuntimeError, match="sample indices") as captured:
        list(loader)
    assert isinstance(captured.value.__cause__, OSError)


def test_geometry_transforms_and_validation() -> None:
    image = Image.fromarray(np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3))
    centered = mt.data.transforms.CenterCrop((8, 10))(image)
    assert centered.size == (10, 8)
    cropped = mt.data.transforms.RandomCrop((4, 5), padding=1)(image)
    assert cropped.size == (5, 4)
    nearest = mt.data.transforms.Resize(3, interpolation="nearest")(image)
    assert nearest.size == (3, 3)
    flipped = mt.data.transforms.RandomHorizontalFlip(1.0)(np.asarray(image))
    np.testing.assert_array_equal(flipped[:, 0], np.asarray(image)[:, -1])
    grayscale = mt.data.transforms.ToArray()(np.zeros((2, 3), dtype=np.uint8))
    assert grayscale.shape == (1, 2, 3)
    with pytest.raises(ValueError, match="exceeds"):
        mt.data.transforms.RandomCrop((20, 20))(image)
    with pytest.raises(ValueError, match="matching"):
        mt.data.transforms.Normalize([0, 0], [1])


def test_sampler_state_replay_and_loader_conflicts() -> None:
    source = list(range(6))
    sampler = mt.data.RandomSampler(source, replacement=True, num_samples=8, seed=7)
    list(sampler)
    state = sampler.state_dict()
    expected = list(sampler)
    restored = mt.data.RandomSampler(source, replacement=True, num_samples=8, seed=9)
    restored.load_state_dict(state)
    assert list(restored) == expected

    subset = mt.data.SubsetRandomSampler([1, 3, 5], seed=2)
    assert sorted(subset) == [1, 3, 5]
    weights = mt.data.WeightedRandomSampler([1, 2, 3], 4, seed=4)
    list(weights)
    weight_state = weights.state_dict()
    expected = list(weights)
    clone = mt.data.WeightedRandomSampler([1, 2, 3], 4, seed=99)
    clone.load_state_dict(weight_state)
    assert list(clone) == expected

    dataset = mt.data.TensorDataset(mt.arange(6))
    with pytest.raises(ValueError, match="shuffle"):
        mt.data.DataLoader(
            dataset, shuffle=True, sampler=mt.data.SequentialSampler(dataset)
        )
    batches = mt.data.BatchSampler(mt.data.SequentialSampler(dataset), 2, False)
    with pytest.raises(ValueError, match="conflicts"):
        mt.data.DataLoader(dataset, batch_size=2, batch_sampler=batches)
    with pytest.raises(ValueError, match="requires"):
        mt.data.DataLoader(dataset, persistent_workers=True)


def test_csv_without_header_and_dataset_errors(tmp_path) -> None:
    path = tmp_path / "plain.csv"
    path.write_text("1;2;0\n3;4;1\n", encoding="utf-8")
    dataset = mt.data.CSVDataset(
        path,
        target_column=2,
        feature_columns=[0, 1],
        delimiter=";",
        has_header=False,
        target_dtype=np.float32,
    )
    assert dataset[-1][1] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="string columns"):
        mt.data.CSVDataset(path, target_column="label", has_header=False, delimiter=";")
    bad = tmp_path / "bad.csv"
    bad.write_text("x,label\nnot-a-number,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid numeric"):
        mt.data.CSVDataset(bad, target_column="label")[0]
