from std.bit import next_power_of_two
from data_traits import HasData, HasLabel

comptime chars = " .,:-=+*#%@"
comptime extended_chars = (
    " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
)


@inline(.always)
def print_grayscale[
    dtype: DType, //, charset: String = chars
](intensity: Scalar[dtype], mut w: Some[Writer], max_val: Float32 = 255.0):
    comptime charset_max = Float32(charset.byte_length() - 1)
    var pct = (intensity.cast[DType.float32]() / max_val).clamp(0.0, 1.0)
    var index = Int(pct * charset_max)
    w.write(charset[byte=index])


def render_ascii_image(
    pixels: Span[Float32, _],
    rows: Int = 28,
    cols: Int = 28,
    normalized: Bool = True,
) -> String:
    """Renders a grayscale image (row-major) as an ASCII art string."""
    var out = String()
    var scale: Float32 = 1.0 if normalized else 255.0
    for r in range(rows):
        for c in range(cols):
            print_grayscale[chars](pixels[r * cols + c], out, max_val=scale)
        out.write("\n")
    return out^


comptime TrainImage = TrainData[type=DType.uint8, ...]


struct TrainData[rows: Int, cols: Int, type: DType](
    HasData, HasLabel, TrivialRegisterPassable, Writable
):
    comptime dtype = Self.type
    comptime size = Self.cols * Self.rows
    comptime simd_size = next_power_of_two(Self.size)
    comptime Data = SIMD[Self.dtype, Self.simd_size]

    var label: UInt8
    var data: Self.Data

    def __init__(
        out self,
        data: StringSpan,
        init: Int,
        end: Int,
    ) raises:
        var row_span = data[byte=init:end]
        var bytes = row_span.as_bytes()
        var pos = 0
        var n = len(bytes)

        # Parse label
        var lbl = 0
        while pos < n and bytes[pos] != UInt8(ord(",")):
            var b = bytes[pos]
            if b >= UInt8(ord("0")) and b <= UInt8(ord("9")):
                lbl = lbl * 10 + Int(b - UInt8(ord("0")))
            pos += 1
        self.label = UInt8(lbl)
        pos += 1  # skip ','

        var final_data = Self.Data(0)
        var pixel_idx = 0
        var val = 0
        var has_digit = False
        while pos < n and pixel_idx < Self.size:
            var b = bytes[pos]
            if b >= UInt8(ord("0")) and b <= UInt8(ord("9")):
                val = val * 10 + Int(b - UInt8(ord("0")))
                has_digit = True
            elif b == UInt8(ord(",")):
                if has_digit:
                    final_data[pixel_idx] = Scalar[Self.dtype](val)
                pixel_idx += 1
                val = 0
                has_digit = False
            pos += 1
        if has_digit and pixel_idx < Self.size:
            final_data[pixel_idx] = Scalar[Self.dtype](val)

        self.data = final_data

    @inline(.always)
    def get_data(self) -> Self.Data:
        return self.data

    @inline(.always)
    def get_label(self) -> Int:
        return Int(self.label)

    def write_to(self, mut writer: Some[Writer]):
        for r in range(Self.rows):
            for c in range(Self.cols):
                print_grayscale[extended_chars](
                    self.data[r * Self.cols + c], writer
                )
            writer.write("\n")

    @inline(.always)
    def __len__(self) -> Int:
        return Self.size

    @inline(.always)
    def __getitem__(self, x: Int, y: Int) -> Scalar[Self.dtype]:
        return self.data[x + y * Self.cols]

    @inline(.always)
    def __getitem__(self, i: Int) -> Scalar[Self.dtype]:
        return self.data[i]
