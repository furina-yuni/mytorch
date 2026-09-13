"""Train a mini-batch GPU classifier and save its state."""

import mytorch as mt

DEVICE = "cuda:0"
CHECKPOINT = "classifier.npz"


class Classifier(mt.nn.Module):
    def __init__(self, device: str = DEVICE) -> None:
        super().__init__()
        self.network = mt.nn.Sequential(
            mt.nn.Linear(2, 16, device=device),
            mt.nn.ReLU(),
            mt.nn.Linear(16, 3, device=device),
        )

    def forward(self, inputs: mt.Tensor) -> mt.Tensor:
        # A classification model returns logits, not probabilities.
        return self.network(inputs)


def make_data() -> tuple[mt.data.TensorDataset, mt.Tensor, mt.Tensor]:
    # Python values move to the selected GPU when tensor() is called.
    train_x = mt.tensor(
        [
            [-2.4, -1.2],
            [-2.0, -0.8],
            [-1.8, -1.4],
            [-2.2, -0.5],
            [2.4, -1.2],
            [2.0, -0.8],
            [1.8, -1.4],
            [2.2, -0.5],
            [-0.4, 2.2],
            [0.0, 1.7],
            [0.5, 2.4],
            [-0.6, 1.5],
        ],
        dtype=mt.float32,
        device=DEVICE,
    )
    train_y = mt.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
        dtype=mt.int64,
        device=DEVICE,
    )
    valid_x = mt.tensor(
        [
            [-2.1, -1.0],
            [-1.7, -0.7],
            [2.1, -1.0],
            [1.7, -0.7],
            [0.0, 2.1],
            [0.4, 1.8],
        ],
        dtype=mt.float32,
        device=DEVICE,
    )
    valid_y = mt.tensor([0, 0, 1, 1, 2, 2], dtype=mt.int64, device=DEVICE)
    return mt.data.TensorDataset(train_x, train_y), valid_x, valid_y


def accuracy(logits: mt.Tensor, target: mt.Tensor) -> float:
    predicted = logits.argmax(dim=1)
    # Copy only the final scalar to CPU for display.
    return float((predicted == target).mean().item())


def main() -> None:
    if not mt.cuda.is_available():
        raise RuntimeError("MyTorch training requires a CUDA GPU")

    mt.manual_seed(2026)
    train_dataset, valid_x, valid_y = make_data()
    train_loader = mt.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        seed=2026,
    )
    model = Classifier()
    criterion = mt.nn.CrossEntropyLoss()
    optimizer = mt.optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-4)

    for epoch in range(1, 301):
        model.train()
        for batch_x, batch_y in train_loader:
            # Clear -> forward -> loss -> backward -> update.
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        if epoch == 1 or epoch % 50 == 0:
            model.eval()
            with mt.no_grad():
                valid_logits = model(valid_x)
                valid_accuracy = accuracy(valid_logits, valid_y)
            print(
                f"epoch={epoch:3d} loss={loss.item():.6f} "
                f"valid_accuracy={valid_accuracy:.1%}"
            )

    model.eval()
    mt.save(model.state_dict(), CHECKPOINT)
    print(f"saved: {CHECKPOINT}")


if __name__ == "__main__":
    main()
