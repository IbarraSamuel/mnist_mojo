"""Unit tests for the isolated MAX Python MNIST implementation."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from max.driver import CPU
from max.dtype import DType
from max.graph import TensorType
from max.tensor import Tensor
import numpy as np

from python_mnist.data import augment_shift_batch, preprocess_image_for_mnist
from python_mnist.model import (
    HIDDEN1,
    HIDDEN2,
    IMG_PIXELS,
    NUM_CLASSES,
    MnistEvalWrapper,
    MnistMLP,
    compile_train_step,
    init_he_state,
    load_mlp_for_inference,
    save_weights_npz,
    tensor_scalar,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIGIT = REPO_ROOT / "examples" / "digit_2.png"


class TestMaxMnist(unittest.TestCase):
    def test_train_step_reduces_loss(self) -> None:
        dev = CPU()
        batch_size = 16
        step_fn = compile_train_step(batch_size, dev)
        state = init_he_state(dev, seed=7)

        rng = np.random.default_rng(7)
        x_np = rng.random((batch_size, IMG_PIXELS), dtype=np.float32)
        y_idx_np = (np.arange(batch_size) % NUM_CLASSES).astype(np.int64)
        y_smooth_np = np.zeros((batch_size, NUM_CLASSES), dtype=np.float32)
        y_smooth_np[np.arange(batch_size), y_idx_np] = 1.0

        ones1 = np.ones((batch_size, HIDDEN1), dtype=np.float32)
        ones2 = np.ones((batch_size, HIDDEN2), dtype=np.float32)

        losses = []
        for s in range(1, 6):
            out = step_fn(
                Tensor.from_dlpack(x_np).to(dev),
                Tensor.from_dlpack(y_smooth_np).to(dev),
                Tensor.from_dlpack(y_idx_np).to(dev),
                Tensor.from_dlpack(ones1).to(dev),
                Tensor.from_dlpack(ones2).to(dev),
                Tensor(1e-3, dtype=DType.float32, device=dev),
                Tensor(1.0 / (1.0 - 0.9**s), dtype=DType.float32, device=dev),
                Tensor(1.0 / (1.0 - 0.999**s), dtype=DType.float32, device=dev),
                *state,
            )
            losses.append(tensor_scalar(out[0]))
            state = list(out[2:])

        self.assertGreater(losses[0], 0.5)
        self.assertLess(losses[-1], losses[0])

    def test_weights_roundtrip_and_inference(self) -> None:
        dev = CPU()
        state = init_he_state(dev, seed=11)
        params = state[:6]

        with tempfile.TemporaryDirectory() as tmp:
            wpath = Path(tmp) / "weights.npz"
            save_weights_npz(str(wpath), params)

            eval_fn = load_mlp_for_inference(str(wpath), batch_size=4, device=dev)
            x_np = np.ones((4, IMG_PIXELS), dtype=np.float32) * 0.25
            logits_t, preds_t = eval_fn(Tensor.from_dlpack(x_np).to(dev))
            logits = np.from_dlpack(logits_t.to(CPU()))
            preds = np.from_dlpack(preds_t.to(CPU()))

            self.assertEqual(logits.shape, (4, NUM_CLASSES))
            self.assertEqual(preds.shape, (4, 1))

    def test_augment_and_preprocess_sample(self) -> None:
        rng = np.random.default_rng(1)
        x = np.ones((8, IMG_PIXELS), dtype=np.float32)
        shifted = augment_shift_batch(x, rng)
        self.assertEqual(shifted.shape, (8, IMG_PIXELS))

        if SAMPLE_DIGIT.exists():
            arr, (w, h) = preprocess_image_for_mnist(SAMPLE_DIGIT)
            self.assertEqual(arr.shape, (1, IMG_PIXELS))
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)
            self.assertGreater(float(arr.max()), 0.5)


if __name__ == "__main__":
    unittest.main()
