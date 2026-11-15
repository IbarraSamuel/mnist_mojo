# `python_mnist` — MNIST Training & Inference with MAX

Isolated Python + MAX implementation using `uv`, `max.nn`, `max.functional`, and `max.tensor`.

## Architecture & Improvements

- **Model**: `784 -> 512 -> 256 -> 10` MLP (`max.nn.Linear` + `F.relu`)
- **Compiled Training Graph (`TrainStep`)**: Fused forward pass, softmax, label-smoothed cross-entropy (`0.05`), backward pass, and **AdamW** updates (`lr` cosine-annealed `2e-3 -> 1e-4`, `weight_decay=1e-4`) compiled into a single MAX graph.
- **Regularization**: Inverted dropout (`0.2`) and random `+/-1` pixel shift augmentation during training.
- **Compiled Inference (`MnistMLP`)**: Uses `max.nn.Module.compile()` for validation, test-set prediction (`predictions.csv`), and single-image CLI inference.

## Commands

```bash
cd python_mnist

# Run unit tests
uv run python -m unittest discover -s tests

# Train and evaluate (uses GPU automatically if available; add --device cpu to force CPU)
uv run train

# On macOS Apple Silicon GPU (if Metal toolchain env var is needed):
TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.922.4 uv run train --device gpu

# Predict a digit image
uv run predict ../examples/digit_2.png
```
