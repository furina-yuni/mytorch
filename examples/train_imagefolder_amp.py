"""Train an ImageFolder classifier with AMP, scheduling, and checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import mytorch as mt


class ImageClassifier(mt.nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.features = mt.nn.Sequential(
            mt.nn.Conv2d(3, 16, 3, padding=1),
            mt.nn.ReLU(),
            mt.nn.MaxPool2d(2),
            mt.nn.Conv2d(16, 32, 3, padding=1),
            mt.nn.ReLU(),
            mt.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = mt.nn.Linear(32, classes)

    def forward(self, input: mt.Tensor) -> mt.Tensor:
        return self.classifier(self.features(input).flatten(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="root/class_name/image directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--checkpoint", type=Path, default=Path("image-training.mtz"))
    args = parser.parse_args()

    mt.manual_seed(2026)
    transform = mt.data.transforms.Compose(
        [
            mt.data.transforms.Resize((32, 32)),
            mt.data.transforms.RandomHorizontalFlip(),
            mt.data.transforms.ToArray(),
            mt.data.transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    dataset = mt.data.ImageFolder(args.data, transform=transform)
    loader = mt.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
        seed=2026,
    )
    model = ImageClassifier(len(dataset.classes))
    optimizer = mt.optim.AdamW(model.parameters(), lr=3e-3)
    scheduler = mt.optim.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = mt.amp.GradScaler()
    criterion = mt.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        total = 0.0
        for images, labels in loader:
            optimizer.zero_grad()
            with mt.amp.autocast():
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        scheduler.step()
        print(f"epoch={epoch + 1} loss={total / len(loader):.6f}")

        mt.save_checkpoint(
            {
                "epoch": epoch + 1,
                "classes": dataset.classes,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "dataloader": loader.state_dict(),
                "rng_state": mt.get_rng_state(),
            },
            args.checkpoint,
        )
    loader.close()


if __name__ == "__main__":
    main()
