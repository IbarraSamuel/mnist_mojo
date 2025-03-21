from pathlib import Path
import os

from memory import Span
from algorithm import sync_parallelize
from layout import Layout, LayoutTensor, composition
from gpu.host import DeviceContext, DeviceBuffer
from gpu import thread_idx, block_idx, barrier
from algorithm import sync_parallelize
from gpu.tensor_ops import mma
import random

alias file_rows = 42000
alias img_rows = 28
alias img_cols = 28
alias img_pixels = img_rows * img_cols

alias num_offset = ord("0")
alias linebreak_byte: UInt8 = ord("\n")
alias chars = " .,:-=+*#%@"
alias extended_chars = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
alias grads = List[String](" ", "░", "▒", "▓", "█")


fn print_grayscale[
    W: Writer, //, charset: List[String] = grads
](owned intensity: Float32, mut w: W):
    alias charset_size = len(charset)
    pct = intensity.cast[DType.float32]() / 255.0
    index = (pct * (charset_size - 1)).cast[DType.uint8]()
    w.write(charset[index])


fn print_grayscale[
    W: Writer, //, charset: String = chars
](owned intensity: Float32, mut w: W):
    alias charset_size = len(charset)
    pct = intensity.cast[DType.float32]() / 255.0
    index = (pct * (charset_size - 1)).cast[DType.uint8]()
    w.write(charset[index])


fn print_grayscale[W: Writer, //](owned intensity: Float32, mut w: W):
    gray_code = 232 + Int(intensity.cast[DType.float32]() * 23.0 / 255.0)
    w.write("\033[48;5;", gray_code, "m")


struct TrainImage(Writable):
    var label: UInt8
    var data: List[Float32]

    fn __init__(
        out self,
        data: Span[UInt8],
        init: Int,
        end: Int,
    ) raises:
        # Slice the span
        self_data = data[init:end]
        string = String(buffer=List(self_data))
        init_pos = string.find(",")
        self.label = Int(string[:init_pos])
        elems = (string^)[init_pos + 1 :].split(",")
        nums_list = List[Float32]()
        for elem in elems:
            nums_list.append(Int(elem[]))

        self.data = nums_list

    fn __copyinit__(out self, other: Self):
        self.label = other.label
        self.data = other.data

    fn __moveinit__(out self, owned other: Self):
        self.label = other.label
        self.data = other.data

    fn write_to[w: Writer](self, mut writer: w):
        for i in range(img_pixels):
            print_grayscale[extended_chars](self.data[i], writer)
            if i % img_cols == 0:
                writer.write("\n")
        writer.write("\n")

    fn __len__(self) -> Int:
        return len(self.data)

    fn __getitem__(self, x: Int, y: Int) -> ref [self.data] Float32:
        return self.data[x + y * img_cols]

    fn __getitem__(self, i: Int) -> ref [self.data] Float32:
        return self.data[i]


fn arange_data(train_data_string: String) raises -> List[TrainImage]:
    jumps = List[Int]()
    n = 0
    while n < len(train_data_string):
        next_jump = train_data_string.find("\n", n)
        if next_jump == -1:
            break

        jumps.append(next_jump)
        n = next_jump + 1

    print("Building images...")

    train_bytes = train_data_string.as_bytes()

    frst = TrainImage(data=train_bytes, init=jumps[0], end=jumps[1])
    images = List[TrainImage]()

    # Fill with something
    for _ in range(file_rows):
        images.append(frst)

    # Then, fill it with multiple threads
    @parameter
    fn calc_each(i: Int) raises:
        img = TrainImage(
            data=train_bytes,
            init=jumps[i + 1],
            end=jumps[i + 2],
        )
        images[i + 1] = img

    sync_parallelize[calc_each](len(jumps) - 2)

    return images


fn dot[
    Mrows: Int,
    Mcols: Int,
    Ncols: Int,
    dtype: DType,
](
    M: LayoutTensor[dtype, Layout.row_major(Mrows, Mcols), MutableAnyOrigin],
    X: LayoutTensor[dtype, Layout.row_major(Mcols, Ncols), MutableAnyOrigin],
    B: LayoutTensor[dtype, Layout.row_major(Mrows, 1), MutableAnyOrigin],
    O: LayoutTensor[dtype, Layout.row_major(Mrows, Ncols), MutableAnyOrigin],
):
    # We will try to do w1
    i = thread_idx.x  # should be w1.shape[1]
    j = thread_idx.y  # should be w1.shape[0]
    k = thread_idx.z  # should be b1.shape[0]
    O[i, k] += M[i, j] * X[j, k] + B[i, 1]


fn main() raises:
    alias dtype = DType.float32
    var data_folder = Path("digit-recognizer")
    test = data_folder / "test.csv"
    train = data_folder / "train.csv"

    _test_data = test.read_text()
    train_data_string = train.read_text()

    images = arange_data(train_data_string)

    # Create a device context to use a gpu.
    ctx = DeviceContext()
    xt = enqueue_build_tensor_from[file_rows, img_pixels](ctx, images)
    w1 = enqueue_build_random_tensor[10, img_pixels](ctx)
    b1 = enqueue_build_random_tensor[10, 1](ctx)
    w2 = enqueue_build_random_tensor[10, 10](ctx)
    b2 = enqueue_build_random_tensor[10, 1](ctx)
    o = enqueue_build_random_tensor[img_pixels, img_pixels](ctx)

    x = enqueue_build_random_tensor[img_pixels, file_rows](ctx)
    ctx.synchronize()

    fn transpose[
        a: Int, b: Int
    ](
        t: LayoutTensor[dtype, Layout.row_major(a, b), MutableAnyOrigin],
        o: LayoutTensor[dtype, Layout.row_major(b, a), MutableAnyOrigin],
    ):
        pass
        o[block_idx.y, block_idx.x] = t[block_idx.x, block_idx.y]

    ctx.enqueue_function[transpose[file_rows, img_pixels]](
        xt, x, grid_dim=(file_rows, img_pixels), block_dim=1
    )
    # transpose[file_rows, img_pixels](xt, x)

    ctx.synchronize()

    fn dot[
        Mrows: Int,
        Mcols: Int,
        Ncols: Int,
        dtype: DType,
    ](
        M: LayoutTensor[
            dtype, Layout.row_major(Mrows, Mcols), MutableAnyOrigin
        ],
        X: LayoutTensor[dtype, Layout.row_major(Mrows, 1), MutableAnyOrigin],
        B: LayoutTensor[dtype, Layout.row_major(Mrows, 1), MutableAnyOrigin],
        O: LayoutTensor[
            dtype, Layout.row_major(Mrows, Ncols), MutableAnyOrigin
        ],
    ):
        # We will try to do w1
        i = thread_idx.x  # should be w1.shape[1]
        j = thread_idx.y  # should be w1.shape[0]
        k = thread_idx.z  # should be b1.shape[0]
        O[i, k] += M[i, j] * X[j, k] + B[i, 1]


fn enqueue_build_tensor_from[
    l: Int, z: Int
](ctx: DeviceContext, images: List[TrainImage]) raises -> LayoutTensor[
    DType.float32, Layout.row_major(l, z), MutableAnyOrigin
]:
    alias dtype = DType.float32
    buf = enqueue_create_buf[dtype](ctx, l, z)
    local_buf = ctx.enqueue_create_host_buffer[dtype](l * z)

    @parameter
    fn load_images(i: Int) raises:
        sub_buf = local_buf.create_sub_buffer[dtype](img_pixels * i, img_pixels)
        ctx.enqueue_copy[dtype](dst_buf=sub_buf, src_ptr=images[i].data.data)

    sync_parallelize[load_images](len(images))

    buf.enqueue_copy_from(local_buf)
    return LayoutTensor[DType.float32, Layout.row_major(l, z)](buf)


fn enqueue_create_buf[
    dtype: DType
](ctx: DeviceContext, l: Int, size: Int) raises -> DeviceBuffer[dtype]:
    buf = ctx.enqueue_create_buffer[dtype](l * size)
    return buf.enqueue_fill(0.0)


fn enqueue_build_random_tensor[
    l: Int, z: Int, /, dtype: DType = DType.float32
](ctx: DeviceContext) raises -> LayoutTensor[
    dtype, Layout.row_major(l, z), MutableAnyOrigin
]:
    var b = enqueue_create_buf[dtype](ctx, l, z)
    enqueue_randomize(ctx, b)
    t = enqueue_buf_to_tensor[Layout.row_major(l, z)](ctx, b)
    return t


fn enqueue_randomize(ctx: DeviceContext, device_buffer: DeviceBuffer) raises:
    var bufflen = len(device_buffer)
    hostbuff = ctx.enqueue_create_host_buffer[device_buffer.type](bufflen)
    random.rand(hostbuff.unsafe_ptr(), len(hostbuff))
    device_buffer.enqueue_copy_from(hostbuff)


fn enqueue_buf_to_tensor[
    dtype: DType, //, layout: Layout
](ctx: DeviceContext, b: DeviceBuffer[dtype]) -> LayoutTensor[
    dtype, layout, b.origin
]:
    return LayoutTensor[dtype, layout](b)
