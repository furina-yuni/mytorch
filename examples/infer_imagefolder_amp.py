"""Run one ImageFolder batch through a saved AMP-trained classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

from train_imagefolder_amp import ImageClassifier

import mytorch as mt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    state = mt.load_checkpoint(args.checkpoint)
    classes = list(state["classes"])
    transform = mt.data.transforms.Compose(
        [
            mt.data.transforms.Resize((32, 32)),
            mt.data.transforms.ToArray(),
            mt.data.transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    dataset = mt.data.ImageFolder(args.data, transform=transform)
    loader = mt.data.DataLoader(dataset, batch_size=32, shuffle=False)
    model = ImageClassifier(len(classes))
    model.load_state_dict(state["model"])
    model.eval()

    images, labels = next(iter(loader))
    with mt.no_grad(), mt.amp.autocast():
        predictions = model(images).argmax(dim=1)
    for predicted, expected in zip(
        predictions.numpy().tolist(), labels.numpy().tolist(), strict=True
    ):
        print(f"prediction={classes[predicted]} target={classes[expected]}")


if __name__ == "__main__":
    main()
