"""MNIST CSV loading, image preprocessing, augmentation, and ASCII preview."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image

ASCII_RAMP = " .,-:;=+*#%@"


def load_mnist_csv(path: str | Path, has_label: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Loads Kaggle MNIST CSV into normalized float32 `(N, 784)` and optional int64 labels."""
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    if has_label:
        labels = raw[:, 0].astype(np.int64)
        pixels = raw[:, 1:] / np.float32(255.0)
        return np.ascontiguousarray(pixels), labels
    pixels = raw / np.float32(255.0)
    return np.ascontiguousarray(pixels), None


def augment_shift_batch(images: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Applies random +/-1 pixel x/y shifts to a `(B, 784)` float32 batch."""
    batch_size = images.shape[0]
    imgs = images.reshape(batch_size, 28, 28)
    out = np.zeros_like(imgs)
    shifts_y = rng.integers(-1, 2, size=batch_size)
    shifts_x = rng.integers(-1, 2, size=batch_size)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            mask = (shifts_y == dy) & (shifts_x == dx)
            if not np.any(mask):
                continue
            src_y0 = max(0, -dy)
            src_y1 = min(28, 28 - dy)
            dst_y0 = max(0, dy)
            dst_y1 = min(28, 28 + dy)
            src_x0 = max(0, -dx)
            src_x1 = min(28, 28 - dx)
            dst_x0 = max(0, dx)
            dst_x1 = min(28, 28 + dx)
            out[mask, dst_y0:dst_y1, dst_x0:dst_x1] = imgs[mask, src_y0:src_y1, src_x0:src_x1]

    return np.ascontiguousarray(out.reshape(batch_size, 784))


def preprocess_image_for_mnist(image_path: str | Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Loads an image, converts to grayscale, centers/scales to MNIST 28x28 format."""
    with Image.open(image_path) as img:
        orig_size = img.size
        gray = img.convert("L")
        arr = np.asarray(gray, dtype=np.float32) / np.float32(255.0)

    h, w = arr.shape
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if float(np.mean(border)) > 0.5:
        arr = 1.0 - arr

    fg = arr > 0.12
    if np.any(fg):
        ys, xs = np.where(fg)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        crop = arr[y0 : y1 + 1, x0 : x1 + 1]
    else:
        crop = arr

    ch, cw = crop.shape
    scale = 20.0 / float(max(ch, cw))
    new_h = max(1, min(20, int(round(ch * scale))))
    new_w = max(1, min(20, int(round(cw * scale))))

    resized_img = Image.fromarray((crop * 255.0).clip(0, 255).astype(np.uint8), mode="L").resize(
        (new_w, new_h), Image.Resampling.BILINEAR
    )
    resized = np.asarray(resized_img, dtype=np.float32) / np.float32(255.0)

    canvas = np.zeros((28, 28), dtype=np.float32)
    off_y = (28 - new_h) // 2
    off_x = (28 - new_w) // 2
    canvas[off_y : off_y + new_h, off_x : off_x + new_w] = resized

    # Shift center of mass to (14, 14) like original MNIST normalization
    mass = float(canvas.sum())
    if mass > 1e-6:
        yy, xx = np.mgrid[0:28, 0:28]
        cy = float((yy * canvas).sum()) / mass
        cx = float((xx * canvas).sum()) / mass
        sy = int(round(13.5 - cy))
        sx = int(round(13.5 - cx))
        shifted = np.zeros_like(canvas)
        src_y0 = max(0, -sy)
        src_y1 = min(28, 28 - sy)
        dst_y0 = max(0, sy)
        dst_y1 = min(28, 28 + sy)
        src_x0 = max(0, -sx)
        src_x1 = min(28, 28 - sx)
        dst_x0 = max(0, sx)
        dst_x1 = min(28, 28 + sx)
        if src_y1 > src_y0 and src_x1 > src_x0:
            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = canvas[src_y0:src_y1, src_x0:src_x1]
            canvas = shifted

    return np.ascontiguousarray(canvas.reshape(1, 784)), orig_size


def render_ascii(pixels_784: np.ndarray) -> str:
    """Renders a 784-element normalized image as 28x28 ASCII art."""
    img = pixels_784.reshape(28, 28)
    lines = []
    n_ramp = len(ASCII_RAMP) - 1
    for r in range(28):
        row_chars = [ASCII_RAMP[min(n_ramp, max(0, int(round(float(v) * n_ramp))))] for v in img[r]]
        lines.append("".join(row_chars))
    return "\n".join(lines)
