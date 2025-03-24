from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor
from layout.math import sum
from data_traits import HasData

from builtin.builtin_slice import slice
from os import abort
from algorithm import sync_parallelize
from memory import UnsafePointer, memcpy
import random
from image import print_grayscale


fn get_gpu() raises -> DeviceContext:
    return DeviceContext()


fn enqueue_create_buf[
    dtype: DType = DType.float32
](ctx: DeviceContext, size: Int) raises -> DeviceBuffer[dtype]:
    buf = ctx.enqueue_create_buffer[dtype](size)
    return buf.enqueue_fill(0)


fn enqueue_create_host_buf[
    dtype: DType = DType.float32
](ctx: DeviceContext, size: Int) raises -> HostBuffer[dtype]:
    buf = ctx.enqueue_create_host_buffer[dtype](size)
    return buf.enqueue_fill(0)


fn enqueue_host_to_gpu[
    dtype: DType
](ctx: DeviceContext, host_buff: HostBuffer[dtype]) raises -> DeviceBuffer[
    dtype
]:
    gpu_buff = ctx.enqueue_create_buffer[dtype](len(host_buff))
    gpu_buff.enqueue_copy_from(host_buff)
    return gpu_buff


fn enqueue_gpu_to_host[
    dtype: DType
](ctx: DeviceContext, gpu_buff: DeviceBuffer[dtype]) raises -> HostBuffer[
    dtype
]:
    host_buff = ctx.enqueue_create_host_buffer[dtype](len(gpu_buff))
    gpu_buff.enqueue_copy_to(host_buff)
    return host_buff


fn enqueue_buf_to_tensor[
    dtype: DType, //, layout: Layout
](ctx: DeviceContext, b: DeviceBuffer[dtype]) -> LayoutTensor[
    dtype, layout, b.origin
]:
    return LayoutTensor[dtype, layout](b)


fn enqueue_randomize(ctx: DeviceContext, gpu_buffer: DeviceBuffer) raises:
    size = len(gpu_buffer)
    host_buffer = ctx.enqueue_create_host_buffer[gpu_buffer.type](size)
    random.rand(host_buffer.unsafe_ptr(), size)
    gpu_buffer.enqueue_copy_from(host_buffer)


fn enqueue_build_matrix[
    rows: Int,
    cols: Int,
    *,
    randomize: Bool = False,
    dtype: DType = DType.float32,
    layout: Layout = Layout.row_major(rows, cols),
](ctx: DeviceContext) raises -> (
    DeviceBuffer[dtype],
    LayoutTensor[dtype, layout, MutableAnyOrigin],
):
    var b = enqueue_create_buf[dtype](ctx, rows * cols)

    @parameter
    if randomize:
        enqueue_randomize(ctx, b)

    return b, enqueue_buf_to_tensor[layout](ctx, b)


fn enqueue_images_to_gpu_matrix[
    img_type: HasData,
    layout: Layout,
](
    ctx: DeviceContext,
    buff: DeviceBuffer[img_type.dtype],
    tensor: LayoutTensor[img_type.dtype, layout, MutableAnyOrigin],
    images: List[img_type],
) raises:
    local_buff = enqueue_create_host_buf[img_type.dtype](
        ctx, len(buff)  # Doesn't matter right now
    )
    local_buff = local_buff.enqueue_fill(0)
    local_tensor = __type_of(tensor)(local_buff)

    values_len = 0
    values_len = len(images) * img_type.size

    if values_len > len(buff):
        msg = (
            "The size of the data: {} is greater than the size of the"
            " buffer: {}".format(values_len, len(buff))
        )
        abort(msg)

    # TODO: Optimize
    alias pixels: Int = tensor.shape[0]()
    alias images_: Int = tensor.shape[1]()
    for pixel in range(pixels):
        for image in range(images_):
            control = Int(images[image].get_data()[pixel])
            local_tensor[pixel, image] = control


# fn dot[
#     Mrows: Int,
#     Mcols: Int,
#     Ncols: Int,
#     dtype: DType,
# ](
#     M: LayoutTensor[dtype, Layout.row_major(Mrows, Mcols), MutableAnyOrigin],
#     X: LayoutTensor[dtype, Layout.row_major(Mrows, 1), MutableAnyOrigin],
#     B: LayoutTensor[dtype, Layout.row_major(Mrows, 1), MutableAnyOrigin],
#     O: LayoutTensor[dtype, Layout.row_major(Mrows, Ncols), MutableAnyOrigin],
# ):
#     # We will try to do w1
#     i = thread_idx.x  # should be w1.shape[1]
#     j = thread_idx.y  # should be w1.shape[0]
#     k = thread_idx.z  # should be b1.shape[0]
#     O[i, k] += M[i, j] * X[j, k] + B[i, 1]


# fn dot[
#     Mrows: Int,
#     Mcols: Int,
#     Ncols: Int,
#     dtype: DType,
# ](
#     M: LayoutTensor[dtype, Layout.row_major(Mrows, Mcols), MutableAnyOrigin],
#     X: LayoutTensor[dtype, Layout.row_major(Mcols, Ncols), MutableAnyOrigin],
#     B: LayoutTensor[dtype, Layout.row_major(Mrows, 1), MutableAnyOrigin],
#     O: LayoutTensor[dtype, Layout.row_major(Mrows, Ncols), MutableAnyOrigin],
# ):
#     # We will try to do w1
#     i = thread_idx.x  # should be w1.shape[1]
#     j = thread_idx.y  # should be w1.shape[0]
#     k = thread_idx.z  # should be b1.shape[0]
#     O[i, k] += M[i, j] * X[j, k] + B[i, 1]
