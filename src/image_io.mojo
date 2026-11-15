from std.math import floor, max, min, round
from std.os import Process, remove
from std.pathlib import Path
from std.sys import CompilationTarget


comptime INV_255 = Float32(1.0 / 255.0)
comptime LUMA_R = Float32(0.299)
comptime LUMA_G = Float32(0.587)
comptime LUMA_B = Float32(0.114)


struct GrayImage(Movable):
    """Grayscale image in row-major [0.0, 1.0] Float32 pixels."""

    var width: Int
    var height: Int
    var pixels: List[Float32]

    def __init__(out self, width: Int, height: Int, var pixels: List[Float32]):
        self.width = width
        self.height = height
        self.pixels = pixels^


@inline(.always)
def _u16_le(bytes: Span[Byte, _], off: Int) -> Int:
    return Int(bytes[off]) | (Int(bytes[off + 1]) << 8)


@inline(.always)
def _i32_le(bytes: Span[Byte, _], off: Int) -> Int:
    var u = (
        UInt32(bytes[off])
        | (UInt32(bytes[off + 1]) << 8)
        | (UInt32(bytes[off + 2]) << 16)
        | (UInt32(bytes[off + 3]) << 24)
    )
    return Int(Int32(u))


def load_bmp_grayscale(path: String) raises -> GrayImage:
    """Decodes an uncompressed 8-bit, 24-bit, or 32-bit BMP into [0, 1] grayscale.
    """
    var bytes = Path(path).read_bytes()
    var span = Span(bytes)
    if (
        len(bytes) < 54
        or bytes[0] != Byte(ord("B"))
        or bytes[1] != Byte(ord("M"))
    ):
        raise Error(t"Not a valid BMP file: {path}")

    var pixel_offset = _i32_le(span, 10)
    var width = _i32_le(span, 18)
    var raw_height = _i32_le(span, 22)
    var bpp = _u16_le(span, 28)
    var compression = _i32_le(span, 30)

    if width <= 0 or raw_height == 0:
        raise Error(t"Invalid BMP dimensions: {width}x{raw_height}")
    if compression != 0 and compression != 3:
        raise Error(
            t"Unsupported BMP compression {compression} (expected uncompressed)"
        )
    if bpp != 8 and bpp != 24 and bpp != 32:
        raise Error(t"Unsupported BMP bit depth {bpp} (supported: 8, 24, 32)")

    var top_down = raw_height < 0
    var height = -raw_height if top_down else raw_height
    var bytes_per_pixel = bpp // 8
    var row_stride = ((width * bpp + 31) // 32) * 4

    if pixel_offset + row_stride * height > len(bytes):
        raise Error("Truncated BMP pixel data")

    var pixels = List[Float32](length=width * height, fill=Float32(0.0))

    for y in range(height):
        var src_y = y if top_down else (height - 1 - y)
        var row_off = pixel_offset + src_y * row_stride
        for x in range(width):
            var p_off = row_off + x * bytes_per_pixel
            var gray: Float32
            if bpp == 8:
                gray = Float32(bytes[p_off]) * INV_255
            elif bpp == 24:
                var b = Float32(bytes[p_off]) * INV_255
                var g = Float32(bytes[p_off + 1]) * INV_255
                var r = Float32(bytes[p_off + 2]) * INV_255
                gray = LUMA_R * r + LUMA_G * g + LUMA_B * b
            else:
                var b = Float32(bytes[p_off]) * INV_255
                var g = Float32(bytes[p_off + 1]) * INV_255
                var r = Float32(bytes[p_off + 2]) * INV_255
                var a = Float32(bytes[p_off + 3]) * INV_255
                var lum = LUMA_R * r + LUMA_G * g + LUMA_B * b
                # Composite transparent pixels against a white background
                gray = lum * a + (1.0 - a)
            pixels[y * width + x] = gray

    return GrayImage(width, height, pixels^)


@inline(.always)
def _is_ws(b: Byte) -> Bool:
    return (
        b == Byte(ord(" "))
        or b == Byte(ord("\t"))
        or b == Byte(ord("\n"))
        or b == Byte(ord("\r"))
    )


def _next_pgm_token(bytes: Span[Byte, _], mut pos: Int) raises -> String:
    var n = len(bytes)
    while pos < n:
        if _is_ws(bytes[pos]):
            pos += 1
        elif bytes[pos] == Byte(ord("#")):
            while pos < n and bytes[pos] != Byte(ord("\n")):
                pos += 1
        else:
            break
    if pos >= n:
        raise Error("Unexpected EOF while parsing PGM header")
    var start = pos
    while pos < n and not _is_ws(bytes[pos]) and bytes[pos] != Byte(ord("#")):
        pos += 1
    return String(unsafe_from_utf8=bytes[start:pos])


def load_pgm_grayscale(path: String) raises -> GrayImage:
    """Decodes a PGM (P5 binary or P2 ASCII) grayscale image."""
    var bytes = Path(path).read_bytes()
    var span = Span(bytes)
    var pos = 0
    var magic = _next_pgm_token(span, pos)
    if magic != "P5" and magic != "P2":
        raise Error(t"Unsupported PGM magic '{magic}' (expected P5 or P2)")

    var width = Int(_next_pgm_token(span, pos))
    var height = Int(_next_pgm_token(span, pos))
    var max_val = Int(_next_pgm_token(span, pos))
    if width <= 0 or height <= 0 or max_val <= 0:
        raise Error("Invalid PGM dimensions or max_val")

    var pixels = List[Float32](length=width * height, fill=Float32(0.0))
    var inv_max = Float32(1.0) / Float32(max_val)

    if magic == "P5":
        pos += 1  # single whitespace delimiter after max_val
        if pos + width * height > len(bytes):
            raise Error("Truncated P5 PGM pixel data")
        for i in range(width * height):
            pixels[i] = Float32(bytes[pos + i]) * inv_max
    else:
        for i in range(width * height):
            var v = Int(_next_pgm_token(span, pos))
            pixels[i] = Float32(v) * inv_max

    return GrayImage(width, height, pixels^)


def load_image_grayscale(path: String) raises -> GrayImage:
    """Loads an image from BMP/PGM natively, or converts PNG/JPG/etc. via macOS sips.
    """
    var lower = path.lower()
    if lower.endswith(".bmp"):
        return load_bmp_grayscale(path)
    if lower.endswith(".pgm"):
        return load_pgm_grayscale(path)

    comptime if CompilationTarget.is_macos():
        var tmp_bmp = "/tmp/mnist_mojo_input.bmp"
        var args: List[String] = [
            "-s",
            "format",
            "bmp",
            path,
            "--out",
            tmp_bmp,
        ]
        var proc = Process.run("/usr/bin/sips", args)
        var status = proc.wait()
        if status.exit_code.or_else(-1) != 0:
            raise Error(
                t"sips failed to convert '{path}' to BMP (exit code"
                t" {status.exit_code.or_else(-1)})"
            )
        var img = load_bmp_grayscale(tmp_bmp)
        try:
            remove(tmp_bmp)
        except:
            pass
        return img^
    else:
        raise Error(
            t"Unsupported image format for '{path}'. On non-macOS systems,"
            t" pass a .bmp or .pgm image."
        )


def _resize_bilinear(
    src: Span[Float32, _],
    src_w: Int,
    src_h: Int,
    dst_w: Int,
    dst_h: Int,
) -> List[Float32]:
    """Resizes a grayscale image using bilinear/area-friendly sampling."""
    var dst = List[Float32](length=dst_w * dst_h, fill=Float32(0.0))
    var scale_x = Float32(src_w) / Float32(dst_w)
    var scale_y = Float32(src_h) / Float32(dst_h)

    for dy in range(dst_h):
        for dx in range(dst_w):
            # If downsampling strongly (>2x), average over the source box
            if scale_x > 1.5 or scale_y > 1.5:
                var x0 = max(0, Int(floor(Float32(dx) * scale_x)))
                var x1 = min(src_w, Int(floor(Float32(dx + 1) * scale_x)) + 1)
                var y0 = max(0, Int(floor(Float32(dy) * scale_y)))
                var y1 = min(src_h, Int(floor(Float32(dy + 1) * scale_y)) + 1)
                var sum = Float32(0.0)
                var cnt = 0
                for sy in range(y0, y1):
                    for sx in range(x0, x1):
                        sum += src[sy * src_w + sx]
                        cnt += 1
                dst[dy * dst_w + dx] = sum / Float32(max(1, cnt))
            else:
                var gx = (Float32(dx) + 0.5) * scale_x - 0.5
                var gy = (Float32(dy) + 0.5) * scale_y - 0.5
                gx = gx.clamp(Float32(0.0), Float32(src_w - 1))
                gy = gy.clamp(Float32(0.0), Float32(src_h - 1))
                var x0 = Int(floor(gx))
                var y0 = Int(floor(gy))
                var x1 = min(x0 + 1, src_w - 1)
                var y1 = min(y0 + 1, src_h - 1)
                var wx = gx - Float32(x0)
                var wy = gy - Float32(y0)
                var v00 = src[y0 * src_w + x0]
                var v10 = src[y0 * src_w + x1]
                var v01 = src[y1 * src_w + x0]
                var v11 = src[y1 * src_w + x1]
                var top = v00 * (1.0 - wx) + v10 * wx
                var bot = v01 * (1.0 - wx) + v11 * wx
                dst[dy * dst_w + dx] = top * (1.0 - wy) + bot * wy

    return dst^


def preprocess_for_mnist(img: GrayImage) -> List[Float32]:
    """Formats an arbitrary grayscale image to match MNIST 28x28 centering.

    Steps:
    1. Auto-invert if the border is light (black digit on white background).
    2. Normalize contrast and clear faint background noise.
    3. Crop to digit bounding box, scale longest side to 20px, and shift so
       the center of mass sits at (14, 14) inside 28x28.
    """
    var w = img.width
    var h = img.height
    var n = w * h
    var work = List[Float32](length=n, fill=Float32(0.0))

    # Estimate background polarity from image border pixels
    var border_sum = Float32(0.0)
    var border_cnt = 0
    for x in range(w):
        border_sum += img.pixels[x]
        border_sum += img.pixels[(h - 1) * w + x]
        border_cnt += 2
    for y in range(1, h - 1):
        border_sum += img.pixels[y * w]
        border_sum += img.pixels[y * w + (w - 1)]
        border_cnt += 2
    var invert = (border_sum / Float32(max(1, border_cnt))) > 0.5

    var max_v = Float32(0.0)
    for i in range(n):
        var v = (1.0 - img.pixels[i]) if invert else img.pixels[i]
        v = v.clamp(Float32(0.0), Float32(1.0))
        work[i] = v
        if v > max_v:
            max_v = v

    if max_v < 0.05:
        return List[Float32](length=784, fill=Float32(0.0))

    # Contrast stretch + noise threshold
    var min_x = w
    var max_x = -1
    var min_y = h
    var max_y = -1
    var inv_max = Float32(1.0) / max_v
    for y in range(h):
        for x in range(w):
            var v = work[y * w + x] * inv_max
            if v < 0.12:
                v = 0.0
            work[y * w + x] = v
            if v > 0.15:
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y

    if max_x < min_x or max_y < min_y:
        return _resize_bilinear(Span(work), w, h, 28, 28)

    # Crop bounding box
    var box_w = max_x - min_x + 1
    var box_h = max_y - min_y + 1
    var cropped = List[Float32](length=box_w * box_h, fill=Float32(0.0))
    for y in range(box_h):
        for x in range(box_w):
            cropped[y * box_w + x] = work[(min_y + y) * w + (min_x + x)]

    # Scale longest dimension to 20px (MNIST convention)
    var new_w: Int
    var new_h: Int
    if box_w >= box_h:
        new_w = 20
        new_h = max(1, Int(round(Float32(box_h) * 20.0 / Float32(box_w))))
    else:
        new_h = 20
        new_w = max(1, Int(round(Float32(box_w) * 20.0 / Float32(box_h))))

    var scaled = _resize_bilinear(Span(cropped), box_w, box_h, new_w, new_h)

    # Compute center of mass of the scaled 20px digit
    var mass = Float32(0.0)
    var com_x = Float32(0.0)
    var com_y = Float32(0.0)
    for y in range(new_h):
        for x in range(new_w):
            var v = scaled[y * new_w + x]
            mass += v
            com_x += Float32(x) * v
            com_y += Float32(y) * v

    var off_x = (28 - new_w) // 2
    var off_y = (28 - new_h) // 2
    if mass > 0.0:
        off_x = Int(round(13.5 - com_x / mass))
        off_y = Int(round(13.5 - com_y / mass))

    var out = List[Float32](length=784, fill=Float32(0.0))
    for y in range(new_h):
        var dy = y + off_y
        if dy < 0 or dy >= 28:
            continue
        for x in range(new_w):
            var dx = x + off_x
            if dx < 0 or dx >= 28:
                continue
            out[dy * 28 + dx] = scaled[y * new_w + x]

    return out^
