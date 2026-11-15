"""MAX-powered MNIST training, validation, and Kaggle test prediction."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

from max.driver import CPU, Device
from max.dtype import DType
from max.tensor import Tensor
import numpy as np

from python_mnist.data import augment_shift_batch, load_mnist_csv, render_ascii
from python_mnist.model import (
    BETA1,
    BETA2,
    HIDDEN1,
    HIDDEN2,
    IMG_PIXELS,
    NUM_CLASSES,
    compile_eval_step,
    compile_train_step,
    init_he_state,
    save_weights_npz,
    select_device,
    tensor_scalar,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent

TRAIN_CSV = REPO_ROOT / "digit-recognizer" / "train.csv"
TEST_CSV = REPO_ROOT / "digit-recognizer" / "test.csv"
WEIGHTS_OUT = PROJECT_ROOT / "mnist_weights.npz"
PREDICTIONS_OUT = PROJECT_ROOT / "predictions.csv"

TRAIN_SPLIT = 40000
VAL_SPLIT = 2000
TEST_ROWS = 28000

BATCH_SIZE = 200
EVAL_BATCH_SIZE = 2000
DEFAULT_EPOCHS = 25
LR_MAX = 2e-3
LR_MIN = 1e-4
DROPOUT_P = 0.2
LABEL_SMOOTHING = 0.05


def _make_smooth_onehot(labels: np.ndarray, smoothing: float) -> np.ndarray:
    n = labels.shape[0]
    y = np.full((n, NUM_CLASSES), smoothing / float(NUM_CLASSES), dtype=np.float32)
    y[np.arange(n), labels] += 1.0 - smoothing
    return np.ascontiguousarray(y)


def _evaluate_compiled(
    eval_fn,
    params: list[Tensor],
    pixels: np.ndarray,
    labels: np.ndarray | None,
    device: Device,
) -> tuple[int, np.ndarray]:
    n = pixels.shape[0]
    all_preds = np.empty(n, dtype=np.int64)
    total_correct = 0
    for start in range(0, n, EVAL_BATCH_SIZE):
        end = start + EVAL_BATCH_SIZE
        xb = Tensor.from_dlpack(np.ascontiguousarray(pixels[start:end])).to(device)
        _, preds_t = eval_fn(xb, *params)
        preds = np.from_dlpack(preds_t.to(CPU())).reshape(-1)
        all_preds[start:end] = preds
        if labels is not None:
            total_correct += int(np.sum(preds == labels[start:end]))
    return total_correct, all_preds


def write_submission_csv(path: Path, preds: np.ndarray) -> None:
    lines = ["ImageId,Label\n"]
    lines.extend(f"{i + 1},{int(p)}\n" for i, p in enumerate(preds))
    path.write_text("".join(lines))


def train_and_evaluate(device_pref: str = "auto", epochs: int = DEFAULT_EPOCHS) -> None:
    device = select_device(device_pref)
    print(f"Using MAX device: {device}")

    t_load = time.perf_counter()
    train_all_x, train_all_y = load_mnist_csv(TRAIN_CSV, has_label=True)
    test_x, _ = load_mnist_csv(TEST_CSV, has_label=False)
    assert train_all_y is not None
    print(
        f"Loaded {train_all_x.shape[0]} train/val images and {test_x.shape[0]} test images "
        f"in {time.perf_counter() - t_load:.3f}s"
    )

    x_train = train_all_x[:TRAIN_SPLIT]
    y_train = train_all_y[:TRAIN_SPLIT]
    x_val = train_all_x[TRAIN_SPLIT : TRAIN_SPLIT + VAL_SPLIT]
    y_val = train_all_y[TRAIN_SPLIT : TRAIN_SPLIT + VAL_SPLIT]

    y_train_smooth = _make_smooth_onehot(y_train, LABEL_SMOOTHING)

    print("Compiling MAX training and inference graphs...")
    t_comp = time.perf_counter()
    train_step = compile_train_step(BATCH_SIZE, device)
    eval_fn = compile_eval_step(EVAL_BATCH_SIZE, device)
    print(f"Compiled training and inference graphs in {time.perf_counter() - t_comp:.3f}s")

    state = init_he_state(device, seed=42)
    rng = np.random.default_rng(42)

    num_batches = TRAIN_SPLIT // BATCH_SIZE
    total_steps = epochs * num_batches
    step_num = 0
    keep_prob = 1.0 - DROPOUT_P
    inv_keep = np.float32(1.0 / keep_prob)

    best_val_correct = -1
    best_params: list[Tensor] | None = None

    print(
        f"\nTraining MAX MLP ({IMG_PIXELS} -> {HIDDEN1} -> {HIDDEN2} -> {NUM_CLASSES}) "
        f"for {epochs} epochs (batch_size={BATCH_SIZE}, AdamW cosine LR {LR_MAX}->{LR_MIN}, "
        f"dropout={DROPOUT_P}, label_smoothing={LABEL_SMOOTHING})..."
    )

    t_train_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        t_ep = time.perf_counter()
        perm = rng.permutation(TRAIN_SPLIT)
        epoch_loss = 0.0
        epoch_correct = 0

        for b_idx in range(num_batches):
            step_num += 1
            progress = float(step_num - 1) / float(max(1, total_steps - 1))
            lr_val = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + math.cos(math.pi * progress))
            inv_bc1_val = 1.0 / (1.0 - BETA1**step_num)
            inv_bc2_val = 1.0 / (1.0 - BETA2**step_num)

            batch_idx = perm[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
            xb_np = augment_shift_batch(x_train[batch_idx], rng)
            yb_smooth_np = np.ascontiguousarray(y_train_smooth[batch_idx])
            yb_idx_np = np.ascontiguousarray(y_train[batch_idx])

            d1_np = ((rng.random((BATCH_SIZE, HIDDEN1)) < keep_prob).astype(np.float32)) * inv_keep
            d2_np = ((rng.random((BATCH_SIZE, HIDDEN2)) < keep_prob).astype(np.float32)) * inv_keep

            xb = Tensor.from_dlpack(xb_np).to(device)
            yb_smooth = Tensor.from_dlpack(yb_smooth_np).to(device)
            yb_idx = Tensor.from_dlpack(yb_idx_np).to(device)
            d1 = Tensor.from_dlpack(d1_np).to(device)
            d2 = Tensor.from_dlpack(d2_np).to(device)
            lr_t = Tensor(lr_val, dtype=DType.float32, device=device)
            bc1_t = Tensor(inv_bc1_val, dtype=DType.float32, device=device)
            bc2_t = Tensor(inv_bc2_val, dtype=DType.float32, device=device)

            out = train_step(xb, yb_smooth, yb_idx, d1, d2, lr_t, bc1_t, bc2_t, *state)
            loss_t, correct_t = out[0], out[1]
            state = list(out[2:])

            epoch_loss += tensor_scalar(loss_t)
            epoch_correct += int(round(tensor_scalar(correct_t)))

        # Evaluate on validation set via compiled MAX eval graph
        val_correct, val_preds = _evaluate_compiled(eval_fn, state[:6], x_val, y_val, device)

        if val_correct > best_val_correct:
            best_val_correct = val_correct
            best_params = [Tensor.from_dlpack(np.from_dlpack(p.to(CPU())).copy()).to(device) for p in state[:6]]

        avg_loss = epoch_loss / float(num_batches)
        train_acc = 100.0 * float(epoch_correct) / float(TRAIN_SPLIT)
        val_acc = 100.0 * float(val_correct) / float(VAL_SPLIT)
        print(
            f"Epoch {epoch:2d}/{epochs} | loss={avg_loss:.4f} | train_acc={train_acc:.2f}% "
            f"| val_acc={val_acc:.2f}% | time={time.perf_counter() - t_ep:.3f}s"
        )

    total_time = time.perf_counter() - t_train_start
    print(f"\nTraining finished in {total_time:.3f}s!")

    final_params = best_params if best_params is not None else state[:6]
    final_val_correct, final_val_preds = _evaluate_compiled(eval_fn, final_params, x_val, y_val, device)
    print(
        f"Best Validation Accuracy: {final_val_correct}/{VAL_SPLIT} "
        f"({100.0 * float(final_val_correct) / float(VAL_SPLIT):.2f}%)"
    )

    print("\n=== Validation Sample Predictions ===")
    for i in range(4):
        print(f"Val #{i}: predicted={final_val_preds[i]}, actual={y_val[i]}")
        print(render_ascii(x_val[i]))

    _, test_preds = _evaluate_compiled(eval_fn, final_params, test_x, None, device)
    write_submission_csv(PREDICTIONS_OUT, test_preds)
    print(f"\nWrote {TEST_ROWS} test predictions to {PREDICTIONS_OUT}")

    save_weights_npz(str(WEIGHTS_OUT), final_params)
    print(f"Saved trained weights to {WEIGHTS_OUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MNIST MLP using MAX.")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    args = parser.parse_args()
    train_and_evaluate(device_pref=args.device, epochs=args.epochs)


if __name__ == "__main__":
    main()
