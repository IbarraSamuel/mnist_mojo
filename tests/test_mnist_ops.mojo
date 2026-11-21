from std.math import abs
from std.os import remove
from std.testing import assert_equal, assert_true, TestSuite

from cpu_ops import (
    CPUMatrix,
    cpu_add_bias_and_relu,
    cpu_argmax_axis0,
    cpu_matmul_nn,
    cpu_matmul_nt,
    cpu_matmul_tn,
    cpu_relu_backward_mask,
    cpu_softmax_ce_grad,
    cpu_softmax_forward,
    cpu_sum_axis1_scaled,
)
from load_file import _find_line_bounds, _parse_mnist_row
from image_io import load_pgm_grayscale, preprocess_for_mnist
from predict import predict_single
from weights import load_weights, save_weights


def assert_close(
    actual: Float32, expected: Float32, tol: Float32 = 1e-4
) raises:
    assert_true(
        abs(actual - expected) <= tol,
        String(t"expected {expected} +/- {tol}, got {actual}"),
    )


def test_matmul_variants() raises:
    # A: 2x3 = [[1, 2, 3], [4, 5, 6]]
    var a = CPUMatrix(2, 3)
    for i in range(6):
        a.data[i] = Float32(i + 1)

    # B: 3x2 = [[7, 8], [9, 10], [11, 12]]
    var b = CPUMatrix(3, 2)
    for i in range(6):
        b.data[i] = Float32(i + 7)

    # C = A @ B -> [[58, 64], [139, 154]]
    var c = CPUMatrix(2, 2)
    cpu_matmul_nn(a, b, c)
    assert_close(c.data[0], 58.0)
    assert_close(c.data[1], 64.0)
    assert_close(c.data[2], 139.0)
    assert_close(c.data[3], 154.0)

    # A @ A.T -> [[14, 32], [32, 77]]
    var c_nt = CPUMatrix(2, 2)
    cpu_matmul_nt(a, a, c_nt)
    assert_close(c_nt.data[0], 14.0)
    assert_close(c_nt.data[1], 32.0)
    assert_close(c_nt.data[2], 32.0)
    assert_close(c_nt.data[3], 77.0)

    # A.T @ A -> 3x3: [[17, 22, 27], [22, 29, 36], [27, 36, 45]]
    var c_tn = CPUMatrix(3, 3)
    cpu_matmul_tn(a, a, c_tn)
    assert_close(c_tn.data[0], 17.0)
    assert_close(c_tn.data[4], 29.0)
    assert_close(c_tn.data[8], 45.0)


def test_relu_and_bias() raises:
    var z = CPUMatrix(2, 2)
    z.data[0] = -2.0
    z.data[1] = 1.0
    z.data[2] = 0.0
    z.data[3] = 3.0

    var b = CPUMatrix(2, 1)
    b.data[0] = 0.5
    b.data[1] = -1.0

    var a = CPUMatrix(2, 2)
    cpu_add_bias_and_relu(z, b, a)
    # Z becomes [[-1.5, 1.5], [-1.0, 2.0]]
    assert_close(z.data[0], -1.5)
    assert_close(z.data[1], 1.5)
    assert_close(z.data[2], -1.0)
    assert_close(z.data[3], 2.0)
    # A becomes [[0.0, 1.5], [0.0, 2.0]]
    assert_close(a.data[0], 0.0)
    assert_close(a.data[1], 1.5)
    assert_close(a.data[2], 0.0)
    assert_close(a.data[3], 2.0)

    var dz = CPUMatrix(2, 2, fill=10.0)
    cpu_relu_backward_mask(z, dz)
    assert_close(dz.data[0], 0.0)
    assert_close(dz.data[1], 10.0)
    assert_close(dz.data[2], 0.0)
    assert_close(dz.data[3], 10.0)


def test_softmax_ce_and_argmax() raises:
    # 3 classes, 2 samples
    var z = CPUMatrix(3, 2)
    z.data[0] = 1.0
    z.data[1] = 0.0
    z.data[2] = 3.0
    z.data[3] = 5.0
    z.data[4] = 2.0
    z.data[5] = 1.0

    var a = CPUMatrix(3, 2)
    cpu_softmax_forward(z, a)

    # Columns sum to 1
    assert_close(a.data[0] + a.data[2] + a.data[4], 1.0)
    assert_close(a.data[1] + a.data[3] + a.data[5], 1.0)

    var preds = List[Int](length=2, fill=0)
    cpu_argmax_axis0(a, preds)
    assert_equal(preds[0], 1)  # z[1, 0] = 3.0 is max
    assert_equal(preds[1], 1)  # z[1, 1] = 5.0 is max

    var labels: List[Int] = [1, 0]
    var dz = CPUMatrix(3, 2)
    var loss = cpu_softmax_ce_grad(a, Span(labels), dz)
    assert_true(loss > 0.0)
    # Sum of gradients in each column is 0
    assert_close(dz.data[0] + dz.data[2] + dz.data[4], 0.0)
    assert_close(dz.data[1] + dz.data[3] + dz.data[5], 0.0)


def test_sum_axis1_scaled() raises:
    var m = CPUMatrix(2, 3)
    for i in range(6):
        m.data[i] = Float32(i + 1)  # row0: 1+2+3=6, row1: 4+5+6=15
    var out = CPUMatrix(2, 1)
    cpu_sum_axis1_scaled(m, 0.5, out)
    assert_close(out.data[0], 3.0)
    assert_close(out.data[1], 7.5)


def test_csv_crlf_parsing() raises:
    var sample = String("7,0,255,128\r\n3,10,20,30\r\n")
    var starts, ends = _find_line_bounds(sample)
    assert_equal(len(starts), 2)

    var pixels = List[Float32](length=6, fill=0.0)
    var labels = List[Int](length=2, fill=0)
    var bytes = sample.as_bytes()
    for i in range(2):
        _parse_mnist_row[has_label=True](
            bytes,
            starts[i],
            ends[i],
            3,
            pixels.unsafe_ptr().unsafe_offset(i * 3),
            labels.unsafe_ptr().unsafe_offset(i),
        )
    assert_equal(labels[0], 7)
    assert_equal(labels[1], 3)
    assert_close(pixels[0], 0.0)
    assert_close(pixels[1], 1.0)
    assert_close(pixels[2], Float32(128.0 / 255.0))
    assert_close(pixels[5], Float32(30.0 / 255.0))


def test_weights_roundtrip() raises:
    var w1 = CPUMatrix(4, 3)
    var b1 = CPUMatrix(4, 1)
    var w2 = CPUMatrix(2, 4)
    var b2 = CPUMatrix(2, 1)
    for i in range(12):
        w1.data[i] = Float32(i) * 0.25 - 1.0
    for i in range(4):
        b1.data[i] = Float32(i) * 0.1
    for i in range(8):
        w2.data[i] = Float32(i) * -0.5
    b2.data[0] = 0.75
    b2.data[1] = -0.25

    var path = "/tmp/mnist_mojo_test_weights.bin"
    save_weights(path, w1, b1, w2, b2)
    var rw1, rb1, rw2, rb2 = load_weights(path)
    try:
        remove(path)
    except:
        pass

    assert_equal(rw1.rows, 4)
    assert_equal(rw1.cols, 3)
    assert_equal(rw2.rows, 2)
    assert_equal(rw2.cols, 4)
    for i in range(12):
        assert_close(rw1.data[i], w1.data[i], 1e-6)
    for i in range(4):
        assert_close(rb1.data[i], b1.data[i], 1e-6)
    for i in range(8):
        assert_close(rw2.data[i], w2.data[i], 1e-6)
    assert_close(rb2.data[0], b2.data[0], 1e-6)
    assert_close(rb2.data[1], b2.data[1], 1e-6)


def test_pgm_preprocess_and_predict() raises:
    # Write a 4x4 ASCII PGM (P2) with light background (255) and dark ink (0) in center
    var pgm_path = "/tmp/mnist_mojo_test_digit.pgm"
    with open(pgm_path, "w") as f:
        f.write(
            "P2\n4 4\n255\n255 255 255 255\n255 0 0 255\n255 0 0 255\n255 255"
            " 255 255\n"
        )

    var img = load_pgm_grayscale(pgm_path)
    try:
        remove(pgm_path)
    except:
        pass
    assert_equal(img.width, 4)
    assert_equal(img.height, 4)
    assert_close(img.pixels[0], 1.0)
    assert_close(img.pixels[5], 0.0)

    var normed = preprocess_for_mnist(img)
    assert_equal(len(normed), 784)
    # Borders should be inverted to dark background (0.0), center should have ink (>0.5)
    assert_close(normed[0], 0.0)
    assert_true(normed[14 * 28 + 14] > 0.5)

    # Tiny 784->2->3 network that strongly activates class 2 on positive center pixels
    var w1 = CPUMatrix(2, 784, fill=0.0)
    var b1 = CPUMatrix(2, 1, fill=0.0)
    var w2 = CPUMatrix(3, 2, fill=0.0)
    var b2 = CPUMatrix(3, 1, fill=0.0)
    w1.data[14 * 28 + 14] = 2.0
    w2.data[2 * 2 + 0] = 5.0
    var pred, probs = predict_single(Span(normed), w1, b1, w2, b2)
    assert_equal(pred, 2)
    assert_true(probs[2] > Float32(0.9))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
