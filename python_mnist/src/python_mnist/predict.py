"""CLI inference for handwritten digits using compiled MAX MnistMLP."""

from __future__ import annotations

import argparse
from pathlib import Path

from max.driver import CPU
import numpy as np

from python_mnist.data import preprocess_image_for_mnist, render_ascii
from python_mnist.model import load_mlp_for_inference, select_device

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = PROJECT_ROOT / "mnist_weights.npz"


def predict_image(
    image_path: str | Path,
    weights_path: str | Path = DEFAULT_WEIGHTS,
    device_pref: str = "auto",
) -> tuple[int, np.ndarray, tuple[int, int], np.ndarray]:
    """Runs compiled MAX inference on a single image."""
    device = select_device(device_pref)
    pixels_2d, orig_size = preprocess_image_for_mnist(image_path)
    eval_fn = load_mlp_for_inference(str(weights_path), batch_size=1, device=device)

    from max.tensor import Tensor

    xb = Tensor.from_dlpack(pixels_2d).to(device)
    logits_t, preds_t = eval_fn(xb)
    logits = np.from_dlpack(logits_t.to(CPU())).reshape(-1)
    digit = int(np.from_dlpack(preds_t.to(CPU())).reshape(-1)[0])

    shifted = logits - np.max(logits)
    exp_l = np.exp(shifted)
    probs = exp_l / np.sum(exp_l)
    return digit, probs, orig_size, pixels_2d.reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a digit image using MAX.")
    parser.add_argument("image_path", help="Path to digit image (.png, .jpg, .bmp, .pgm, ...)")
    parser.add_argument("weights_path", nargs="?", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    args = parser.parse_args()

    digit, probs, (w, h), pixels_flat = predict_image(
        args.image_path, args.weights_path, device_pref=args.device
    )
    print(f"Input image: {args.image_path} ({w}x{h})")
    print(render_ascii(pixels_flat))
    print(f"Predicted digit: {digit} (confidence: {probs[digit] * 100.0:.5f}%)")
    print("Class probabilities:")
    for d in range(len(probs)):
        print(f"  {d}: {probs[d] * 100.0:.6g}%")


if __name__ == "__main__":
    main()
