from pathlib import Path

import numpy as np
from PIL import Image

import mytorch as mt


class TinyImageModel(mt.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = mt.nn.Sequential(
            mt.nn.Conv2d(3, 4, 1),
            mt.nn.ReLU(),
            mt.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = mt.nn.Linear(4, 2)

    def forward(self, input):
        return self.head(self.features(input).flatten(1))


def test_imagefolder_amp_scheduler_training_reduces_loss(tmp_path: Path) -> None:
    for class_name, channel in [("blue", 2), ("red", 0)]:
        directory = tmp_path / class_name
        directory.mkdir()
        for index in range(8):
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            image[..., channel] = 220 + index
            Image.fromarray(image).save(directory / f"{index}.png")

    dataset = mt.data.ImageFolder(
        tmp_path,
        transform=mt.data.transforms.Compose(
            [
                mt.data.transforms.ToArray(),
                mt.data.transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        ),
    )
    loader = mt.data.DataLoader(
        dataset, batch_size=4, shuffle=True, seed=8, num_workers=2
    )
    mt.manual_seed(8)
    model = TinyImageModel()
    optimizer = mt.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0)
    scheduler = mt.optim.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.005)
    scaler = mt.amp.GradScaler(init_scale=128)

    losses = []
    for _ in range(10):
        total = 0.0
        for images, labels in loader:
            optimizer.zero_grad()
            with mt.amp.autocast():
                loss = mt.nn.functional.cross_entropy(model(images), labels)
            scaler.scale(loss).backward()
            assert scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        losses.append(total)
        scheduler.step()

    assert losses[-1] < losses[0] * 0.05
    with mt.no_grad(), mt.amp.autocast():
        images, labels = next(iter(loader))
        accuracy = (model(images).argmax(dim=1) == labels).mean().item()
    assert accuracy == 1.0
