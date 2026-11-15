from std.memory import unsafe_memcpy
from std.pathlib import Path

from cpu_ops import CPUMatrix


comptime WEIGHTS_MAGIC = "MNST"


@inline(.always)
def _append_i32_le(mut buf: List[Byte], val: Int32):
    var u = UInt32(val)
    buf.append(Byte(u & 0xFF))
    buf.append(Byte((u >> 8) & 0xFF))
    buf.append(Byte((u >> 16) & 0xFF))
    buf.append(Byte((u >> 24) & 0xFF))


@inline(.always)
def _read_i32_le(bytes: Span[Byte, _], offset: Int) -> Int32:
    var b0 = UInt32(bytes[offset])
    var b1 = UInt32(bytes[offset + 1]) << 8
    var b2 = UInt32(bytes[offset + 2]) << 16
    var b3 = UInt32(bytes[offset + 3]) << 24
    return Int32(b0 | b1 | b2 | b3)


@inline(.always)
def _append_f32_slice(mut buf: List[Byte], src: List[Float32]):
    var byte_len = len(src) * 4
    var cur = len(buf)
    buf.resize(cur + byte_len, Byte(0))
    unsafe_memcpy(
        dest=buf.unsafe_ptr().unsafe_offset(cur),
        src=src.unsafe_ptr().unsafe_bitcast[Byte](),
        count=byte_len,
    )


@inline(.always)
def _read_f32_slice(
    bytes: Span[Byte, _], offset: Int, count: Int, mut dst: List[Float32]
):
    var byte_len = count * 4
    unsafe_memcpy(
        dest=dst.unsafe_ptr().unsafe_bitcast[Byte](),
        src=bytes.unsafe_ptr().unsafe_offset(offset),
        count=byte_len,
    )


def save_weights(
    path: String,
    w1: CPUMatrix,
    b1: CPUMatrix,
    w2: CPUMatrix,
    b2: CPUMatrix,
) raises:
    """Saves trained MLP weights (W1, b1, W2, b2) to a compact binary file."""
    var buf = List[Byte]()
    for b in WEIGHTS_MAGIC.as_bytes():
        buf.append(b)
    _append_i32_le(buf, Int32(w1.rows))  # hidden_size
    _append_i32_le(buf, Int32(w1.cols))  # input_size
    _append_i32_le(buf, Int32(w2.rows))  # num_classes
    _append_f32_slice(buf, w1.data)
    _append_f32_slice(buf, b1.data)
    _append_f32_slice(buf, w2.data)
    _append_f32_slice(buf, b2.data)
    Path(path).write_bytes(Span(buf))


def load_weights(
    path: String,
) raises -> Tuple[CPUMatrix, CPUMatrix, CPUMatrix, CPUMatrix]:
    """Loads trained MLP weights (W1, b1, W2, b2) from a binary weights file."""
    var bytes = Path(path).read_bytes()
    if len(bytes) < 16:
        raise Error(t"Weights file {path} is too small or corrupted")

    var magic_bytes = WEIGHTS_MAGIC.as_bytes()
    for i in range(4):
        if bytes[i] != magic_bytes[i]:
            raise Error(t"Invalid magic header in weights file {path}")

    var hidden_size = Int(_read_i32_le(Span(bytes), 4))
    var input_size = Int(_read_i32_le(Span(bytes), 8))
    var num_classes = Int(_read_i32_le(Span(bytes), 12))

    var w1_count = hidden_size * input_size
    var b1_count = hidden_size
    var w2_count = num_classes * hidden_size
    var b2_count = num_classes
    var expected_bytes = 16 + (w1_count + b1_count + w2_count + b2_count) * 4
    if len(bytes) != expected_bytes:
        raise Error(
            t"Weights file size mismatch: expected {expected_bytes} bytes, got"
            t" {len(bytes)}"
        )

    var w1 = CPUMatrix(hidden_size, input_size)
    var b1 = CPUMatrix(hidden_size, 1)
    var w2 = CPUMatrix(num_classes, hidden_size)
    var b2 = CPUMatrix(num_classes, 1)

    var off = 16
    var span = Span(bytes)
    _read_f32_slice(span, off, w1_count, w1.data)
    off += w1_count * 4
    _read_f32_slice(span, off, b1_count, b1.data)
    off += b1_count * 4
    _read_f32_slice(span, off, w2_count, w2.data)
    off += w2_count * 4
    _read_f32_slice(span, off, b2_count, b2.data)

    return w1^, b1^, w2^, b2^


comptime WEIGHTS_MAGIC_3 = "MNS3"


def is_mlp3_weights(path: String) raises -> Bool:
    """Returns True if `path` uses the 3-layer MNS3 weights format."""
    var bytes = Path(path).read_bytes()
    if len(bytes) < 4:
        return False
    var magic = WEIGHTS_MAGIC_3.as_bytes()
    for i in range(4):
        if bytes[i] != magic[i]:
            return False
    return True


def save_weights_mlp3(
    path: String,
    w1: CPUMatrix,
    b1: CPUMatrix,
    w2: CPUMatrix,
    b2: CPUMatrix,
    w3: CPUMatrix,
    b3: CPUMatrix,
) raises:
    """Saves 3-layer MLP weights (w1: I x H1, b1: H1, w2: H1 x H2, b2: H2, w3: C x H2, b3: C).
    """
    var buf = List[Byte]()
    for b in WEIGHTS_MAGIC_3.as_bytes():
        buf.append(b)
    _append_i32_le(buf, Int32(w1.rows))  # input_size
    _append_i32_le(buf, Int32(w1.cols))  # hidden1
    _append_i32_le(buf, Int32(w2.cols))  # hidden2
    _append_i32_le(buf, Int32(w3.rows))  # num_classes
    _append_f32_slice(buf, w1.data)
    _append_f32_slice(buf, b1.data)
    _append_f32_slice(buf, w2.data)
    _append_f32_slice(buf, b2.data)
    _append_f32_slice(buf, w3.data)
    _append_f32_slice(buf, b3.data)
    Path(path).write_bytes(Span(buf))


def load_weights_mlp3(
    path: String,
) raises -> Tuple[
    CPUMatrix, CPUMatrix, CPUMatrix, CPUMatrix, CPUMatrix, CPUMatrix
]:
    """Loads 3-layer MLP weights saved by `save_weights_mlp3`."""
    var bytes = Path(path).read_bytes()
    if len(bytes) < 20:
        raise Error(t"Weights file {path} is too small or corrupted")

    var magic_bytes = WEIGHTS_MAGIC_3.as_bytes()
    for i in range(4):
        if bytes[i] != magic_bytes[i]:
            raise Error(t"Invalid MNS3 magic header in weights file {path}")

    var input_size = Int(_read_i32_le(Span(bytes), 4))
    var hidden1 = Int(_read_i32_le(Span(bytes), 8))
    var hidden2 = Int(_read_i32_le(Span(bytes), 12))
    var num_classes = Int(_read_i32_le(Span(bytes), 16))

    var w1_c = input_size * hidden1
    var b1_c = hidden1
    var w2_c = hidden1 * hidden2
    var b2_c = hidden2
    var w3_c = num_classes * hidden2
    var b3_c = num_classes
    var expected = 20 + (w1_c + b1_c + w2_c + b2_c + w3_c + b3_c) * 4
    if len(bytes) != expected:
        raise Error(
            t"MNS3 weights size mismatch: expected {expected}, got {len(bytes)}"
        )

    var w1 = CPUMatrix(input_size, hidden1)
    var b1 = CPUMatrix(hidden1, 1)
    var w2 = CPUMatrix(hidden1, hidden2)
    var b2 = CPUMatrix(hidden2, 1)
    var w3 = CPUMatrix(num_classes, hidden2)
    var b3 = CPUMatrix(num_classes, 1)

    var off = 20
    var span = Span(bytes)
    _read_f32_slice(span, off, w1_c, w1.data)
    off += w1_c * 4
    _read_f32_slice(span, off, b1_c, b1.data)
    off += b1_c * 4
    _read_f32_slice(span, off, w2_c, w2.data)
    off += w2_c * 4
    _read_f32_slice(span, off, b2_c, b2.data)
    off += b2_c * 4
    _read_f32_slice(span, off, w3_c, w3.data)
    off += w3_c * 4
    _read_f32_slice(span, off, b3_c, b3.data)

    return w1^, b1^, w2^, b2^, w3^, b3^
