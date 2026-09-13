"""Restore the trained classifier and predict new samples."""

import mytorch as mt
from mytorch.nn import functional as F

DEVICE = "cuda:0"
CHECKPOINT = "classifier.npz"
CLASS_NAMES = ("left", "right", "top")


class Classifier(mt.nn.Module):
    # Names and shapes must exactly match the training model.
    def __init__(self, device: str = DEVICE) -> None:
        super().__init__()
        self.network = mt.nn.Sequential(
            mt.nn.Linear(2, 16, device=device),
            mt.nn.ReLU(),
            mt.nn.Linear(16, 3, device=device),
        )

    def forward(self, inputs: mt.Tensor) -> mt.Tensor:
        return self.network(inputs)


def main() -> None:
    if not mt.cuda.is_available():
        raise RuntimeError("MyTorch inference requires a CUDA GPU")

    model = Classifier(device=DEVICE)
    state = mt.load(CHECKPOINT, device=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    new_points = mt.tensor(
        [[-2.1, -1.0], [2.1, -1.0], [0.0, 2.1]],
        dtype=mt.float32,
        device=DEVICE,
    )

    with mt.no_grad():
        logits = model(new_points)
        probabilities = F.softmax(logits, dim=1)
        class_indices = probabilities.argmax(dim=1)

    # Copy only user-facing results back to CPU.
    points_cpu = new_points.numpy()
    probabilities_cpu = probabilities.numpy()
    indices_cpu = class_indices.numpy()
    for point, probs, class_index in zip(
        points_cpu, probabilities_cpu, indices_cpu, strict=True
    ):
        label = CLASS_NAMES[int(class_index)]
        confidence = float(probs[int(class_index)])
        print(f"point={point.tolist()} class={label} confidence={confidence:.2%}")


if __name__ == "__main__":
    main()
