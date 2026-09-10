"""Train and quantize the compact evaluator consumed by agent.py.

Input is one `centipawn<TAB>FEN` record per line. Labels may come from a
locally run analysis engine or other permitted offline source; nothing is
downloaded or consulted at runtime. For example:

    uv run python -m analysis.train_nnue labels.tsv --epochs 30

The output defaults to weights/nnue.npz, which harness.package includes and
agent.py validates before enabling. The feature layout matches agent.py:
color (black, white), piece type (pawn through king), then square.
"""

import argparse
from pathlib import Path

import chess
import numpy as np
import torch

FEATURES = 2 * 6 * 64


def parse_records(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    labels: list[float] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        try:
            raw_score, fen = raw.split("\t", 1)
            board = chess.Board(fen)
            score = float(raw_score)
        except (ValueError, chess.InvalidMoveError) as error:
            message = f"{path}:{line_number}: expected centipawn<TAB>FEN ({error})"
            raise SystemExit(message) from error

        features = np.zeros(FEATURES, dtype=np.float32)
        for color in (chess.BLACK, chess.WHITE):
            for piece_type in range(chess.PAWN, chess.KING + 1):
                base = (int(color) * 6 + piece_type - 1) * 64
                for square in chess.scan_forward(board.pieces_mask(piece_type, color)):
                    features[base + square] = 1.0
        rows.append(features)
        labels.append(score)

    if len(rows) < 32:
        raise SystemExit("need at least 32 labelled positions to train a useful model")
    return np.stack(rows), np.asarray(labels, dtype=np.float32)


class BoardPlaneNet(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden = torch.nn.Linear(FEATURES, hidden_size)
        self.output = torch.nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(torch.relu(self.hidden(features))).squeeze(1)


def quantize(model: BoardPlaneNet, blend: int) -> dict[str, np.ndarray]:
    hidden_scale = 64
    output_scale = 256
    hidden = model.hidden
    output = model.output

    # Runtime sums feature weights into int32 hidden accumulators, then forms
    # the output in the same scale: hidden_scale * output_scale.
    input_weights = torch.round(hidden.weight.detach().cpu() * hidden_scale)
    output_weights = torch.round(output.weight.detach().cpu()[0] * output_scale)
    hidden_bias = torch.round(hidden.bias.detach().cpu() * hidden_scale)
    output_bias = torch.round(output.bias.detach().cpu()[0] * hidden_scale * output_scale)

    if input_weights.abs().max() > np.iinfo(np.int16).max:
        raise SystemExit("input weights exceed int16 after quantization; lower the training scale")
    if output_weights.abs().max() > np.iinfo(np.int16).max:
        raise SystemExit("output weights exceed int16 after quantization; lower the training scale")

    hidden_size = hidden_bias.numel()
    feature_weights = np.zeros((2, 7, 64, hidden_size), dtype=np.int16)
    flat = input_weights.to(torch.int16).numpy().T.reshape(2, 6, 64, hidden_size)
    feature_weights[:, 1:] = flat
    return {
        "feature_weights": feature_weights,
        "hidden_bias": hidden_bias.to(torch.int32).numpy(),
        "output_weights": output_weights.to(torch.int16).numpy(),
        "output_bias": np.asarray(int(output_bias.item()), dtype=np.int64),
        "scale": np.asarray(hidden_scale * output_scale, dtype=np.int64),
        "blend": np.asarray(blend, dtype=np.int64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the compact chess evaluator")
    parser.add_argument("labels", type=Path, help="one centipawn<TAB>FEN record per line")
    parser.add_argument("--output", type=Path, default=Path("weights/nnue.npz"))
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--blend", type=int, default=100,
                        help="percent of the neural score to use at runtime (1-100)")
    arguments = parser.parse_args()
    if arguments.hidden_size < 1 or arguments.epochs < 1 or arguments.batch_size < 1:
        raise SystemExit("hidden size, epochs and batch size must be positive")
    if not 1 <= arguments.blend <= 100:
        raise SystemExit("blend must be between 1 and 100")

    torch.set_num_threads(1)
    torch.manual_seed(20260910)
    features, labels = parse_records(arguments.labels)
    permutation = np.random.default_rng(20260910).permutation(len(labels))
    split = max(1, int(len(labels) * 0.9))
    train_indices, validation_indices = permutation[:split], permutation[split:]
    if not len(validation_indices):
        validation_indices = train_indices[-1:]

    model = BoardPlaneNet(arguments.hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate)
    loss_function = torch.nn.HuberLoss(delta=100.0)
    feature_tensor = torch.from_numpy(features)
    label_tensor = torch.from_numpy(labels)

    for epoch in range(1, arguments.epochs + 1):
        model.train()
        shuffled = train_indices[np.random.default_rng(epoch).permutation(len(train_indices))]
        for start in range(0, len(shuffled), arguments.batch_size):
            batch = shuffled[start:start + arguments.batch_size]
            optimizer.zero_grad()
            loss = loss_function(model(feature_tensor[batch]), label_tensor[batch])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_error = torch.mean(torch.abs(
                model(feature_tensor[validation_indices]) - label_tensor[validation_indices],
            )).item()
        print(f"epoch {epoch:3d}: validation MAE {validation_error:.1f} cp")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output, **quantize(model, arguments.blend))
    print(f"wrote {arguments.output} from {len(labels):,} labelled positions")


if __name__ == "__main__":
    main()
