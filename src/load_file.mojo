from std.pathlib import Path
from std.sys import simd_width_of
from max.algorithm import parallelize

from image import TrainData, render_ascii_image


comptime NL_BYTE = UInt8(ord("\n"))
comptime CR_BYTE = UInt8(ord("\r"))
comptime COMMA_BYTE = UInt8(ord(","))
comptime ZERO_BYTE = UInt8(ord("0"))
comptime INV_255 = Float32(1.0 / 255.0)
comptime U8_SIMD_W = simd_width_of[DType.uint8]()


struct MNISTDataset(Movable):
    """Holds parsed MNIST images (normalized to [0, 1]) and optional labels."""

    var num_samples: Int
    var num_features: Int
    var has_labels: Bool
    var pixels: List[Float32]
    var labels: List[Int]

    def __init__(
        out self,
        num_samples: Int,
        num_features: Int,
        has_labels: Bool,
        var pixels: List[Float32],
        var labels: List[Int],
    ):
        self.num_samples = num_samples
        self.num_features = num_features
        self.has_labels = has_labels
        self.pixels = pixels^
        self.labels = labels^

    def ascii_preview(
        self, sample_idx: Int, rows: Int = 28, cols: Int = 28
    ) -> String:
        var start = sample_idx * self.num_features
        var end = start + self.num_features
        return render_ascii_image(
            Span(self.pixels)[start:end], rows=rows, cols=cols, normalized=True
        )


def _find_line_bounds(
    data: StringSpan,
    capacity_hint: Int = 0,
) -> Tuple[List[Int], List[Int]]:
    """Finds (start, end) byte offsets for every non-empty line, stripping \\r.
    """
    var bytes = data.as_bytes()
    var ptr = bytes.unsafe_ptr()
    var n = len(bytes)
    var cap = capacity_hint if capacity_hint > 0 else 64
    var starts = List[Int](capacity=cap)
    var ends = List[Int](capacity=cap)
    var nl_vec = SIMD[DType.uint8, U8_SIMD_W](NL_BYTE)

    var i = 0
    while i < n:
        var line_start = i
        while i + U8_SIMD_W <= n:
            if ptr.unsafe_load[width=U8_SIMD_W](i).eq(nl_vec).reduce_or():
                break
            i += U8_SIMD_W
        while i < n and ptr[unsafe_offset=i] != NL_BYTE:
            i += 1
        var line_end = i
        if line_end > line_start and ptr[unsafe_offset=line_end - 1] == CR_BYTE:
            line_end -= 1
        if line_end > line_start:
            starts.append(line_start)
            ends.append(line_end)
        i += 1
    return starts^, ends^


@inline(.always)
def _parse_mnist_row[
    has_label: Bool
](
    bytes: Span[UInt8, _],
    start: Int,
    end: Int,
    num_features: Int,
    pixel_out: MutPointer[Float32, _],
    label_out: MutPointer[Int, _],
):
    """Parses a single CSV row directly into normalized Float32 pixels."""
    var ptr = bytes.unsafe_ptr()
    var pos = start

    comptime if has_label:
        var lbl = 0
        while pos < end:
            var b = ptr[unsafe_offset=pos]
            if b == COMMA_BYTE:
                break
            var d = b - ZERO_BYTE
            if d <= 9:
                lbl = lbl * 10 + Int(d)
            pos += 1
        label_out[] = lbl
        pos += 1  # skip ','

    var feat_idx = 0
    var val = 0
    var has_digit = False

    while pos < end and feat_idx < num_features:
        var b = ptr[unsafe_offset=pos]
        var d = b - ZERO_BYTE
        if d <= 9:
            val = val * 10 + Int(d)
            has_digit = True
        elif b == COMMA_BYTE:
            pixel_out[unsafe_offset=feat_idx] = Float32(
                val
            ) * INV_255 if has_digit else Float32(0.0)
            feat_idx += 1
            val = 0
            has_digit = False
        pos += 1

    if has_digit and feat_idx < num_features:
        pixel_out[unsafe_offset=feat_idx] = Float32(val) * INV_255


def load_mnist_csv[
    has_label: Bool = True
](
    path: String,
    num_features: Int = 784,
    expected_rows: Int = -1,
) raises -> MNISTDataset:
    """Loads an MNIST CSV file (train.csv or test.csv) in parallel."""
    var text = Path(path).read_text()
    var first_nl = text.find("\n")
    if first_nl == -1:
        raise Error(t"CSV file {path} has no header newline")

    var body = text[byte = first_nl + 1 :]
    var starts, ends = _find_line_bounds(
        body, capacity_hint=expected_rows if expected_rows > 0 else 0
    )
    var num_rows = len(starts)

    if expected_rows != -1 and num_rows != expected_rows:
        raise Error(
            t"Expected {expected_rows} rows in {path}, found {num_rows}"
        )

    var pixels = List[Float32](
        length=num_rows * num_features, fill=Float32(0.0)
    )
    var labels = List[Int](length=num_rows if has_label else 0, fill=0)

    var body_bytes = body.as_bytes()
    var pixel_ptr = pixels.unsafe_ptr()
    var label_ptr = labels.unsafe_ptr()
    var starts_span = Span(starts)
    var ends_span = Span(ends)

    def parse_worker(
        row_idx: Int,
    ) {
        imm body_bytes,
        imm starts_span,
        imm ends_span,
        imm pixel_ptr,
        imm label_ptr,
        imm num_features,
    }:
        var dummy_lbl = 0
        var lbl_p = label_ptr.unsafe_offset(row_idx) if has_label else Pointer(
            to=dummy_lbl
        )
        _parse_mnist_row[has_label](
            body_bytes,
            starts_span[row_idx],
            ends_span[row_idx],
            num_features,
            pixel_ptr.unsafe_offset(row_idx * num_features),
            lbl_p,
        )

    parallelize(parse_worker, num_rows)

    return MNISTDataset(
        num_samples=num_rows,
        num_features=num_features,
        has_labels=has_label,
        pixels=pixels^,
        labels=labels^,
    )


def read_image_file[
    path: String,
    *,
    rows: Int,
    cols: Int,
    dtype: DType,
    size: Int = -1,
]() raises -> List[TrainData[rows, cols, dtype]]:
    """Loads labeled MNIST rows into a List[TrainData] in parallel."""
    comptime TD = TrainData[rows, cols, dtype]
    var text_data = Path(path).read_text()
    var first_nl = text_data.find("\n")
    if first_nl == -1:
        raise Error(t"CSV file {path} has no header newline")

    var body = text_data[byte = first_nl + 1 :]
    var starts, ends = _find_line_bounds(body)
    var file_rows = len(starts)

    if size != -1 and file_rows != size:
        raise Error(t"Wrong sizes, found {file_rows} rows, expected {size}")

    var dummy = TD(data=body, init=starts[0], end=ends[0])
    var images = List[TD](length=file_rows, fill=dummy.copy())
    var img_ptr = images.unsafe_ptr()
    var starts_span = Span(starts)
    var ends_span = Span(ends)

    def worker(i: Int) {imm body, imm starts_span, imm ends_span, imm img_ptr}:
        try:
            img_ptr.unsafe_offset(i).unsafe_write(
                TD(data=body, init=starts_span[i], end=ends_span[i])
            )
        except e:
            print("Failed parsing row", i, ":", e)

    parallelize(worker, file_rows)
    return images^
