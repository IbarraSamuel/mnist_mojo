from std.sys import argv

from cpu_ops import (
    CPUMatrix,
    cpu_predict_mlp_single,
    cpu_predict_mlp3_single,
)
from image import render_ascii_image
from image_io import load_image_grayscale, preprocess_for_mnist
from weights import is_mlp3_weights, load_weights, load_weights_mlp3


comptime DEFAULT_WEIGHTS_PATH = "mnist_weights.bin"


def predict_single(
    pixels_28x28: Span[Float32, _],
    w1: CPUMatrix,
    b1: CPUMatrix,
    w2: CPUMatrix,
    b2: CPUMatrix,
) raises -> Tuple[Int, List[Float32]]:
    """Runs forward inference on a single 784-pixel image."""
    return cpu_predict_mlp_single(pixels_28x28, w1, b1, w2, b2)


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("Usage: pixi run predict <image_path> [weights_path]")
        print("Supports .png, .jpg, .bmp, .pgm, .heic, .tiff, .gif")
        return

    var img_path = String(args[1])
    var weights_path = String(args[2]) if len(args) >= 3 else String(
        DEFAULT_WEIGHTS_PATH
    )

    var raw_img = load_image_grayscale(img_path)
    var mnist_pixels = preprocess_for_mnist(raw_img)

    var digit: Int
    var probs: List[Float32]
    if is_mlp3_weights(weights_path):
        var w1, b1, w2, b2, w3, b3 = load_weights_mlp3(weights_path)
        digit, probs = cpu_predict_mlp3_single(
            Span(mnist_pixels), w1, b1, w2, b2, w3, b3
        )
    else:
        var w1, b1, w2, b2 = load_weights(weights_path)
        digit, probs = predict_single(Span(mnist_pixels), w1, b1, w2, b2)
    var confidence = probs[digit] * 100.0

    print(t"Input image: {img_path} ({raw_img.width}x{raw_img.height})")
    print(
        render_ascii_image(
            Span(mnist_pixels), rows=28, cols=28, normalized=True
        )
    )
    print(t"Predicted digit: {digit} (confidence: {confidence}%)")
    print("Class probabilities:")
    for d in range(len(probs)):
        var pct = probs[d] * 100.0
        print(t"  {d}: {pct}%")
