from gpu.host import DeviceContext, DeviceBuffer
from gpu import barrier
from layout import Layout, LayoutTensor, composition, IntTuple
from gpu.id import thread_idx, block_idx
from bit import next_power_of_two
from gpu_mem import (
    enqueue_create_matrix,
    enqueue_create_host_buf,
    MAX_BLOCKS_1D,
    MAX_BLOCKS_2D,
    MAX_BLOCKS_3D,
)

from math import e


fn print_matrix[
    r: Int = -1, c: Int = -1
](
    ctx: DeviceContext, buff: DeviceBuffer, matrix: LayoutTensor[buff.type]
) raises:
    alias rows = r if r != -1 else matrix.shape[0]()
    alias cols = c if c != -1 else matrix.shape[1]()

    host_buffer = enqueue_create_host_buf[matrix.dtype](ctx, rows * cols)
    buff.enqueue_copy_to(host_buffer)
    ctx.synchronize()

    for i in range(rows * cols):
        print(host_buffer[i], end="")
        if i % cols == 0:
            print()


# fn _test_limits(ctx: DeviceContext) raises:
#     fn print_shape():
#         print(
#             "Blocks: [",
#             block_idx.x,
#             block_idx.y,
#             block_idx.z,
#             "], Threads: [",
#             thread_idx.x,
#             thread_idx.y,
#             thread_idx.z,
#             "]",
#         )

#     ctx.enqueue_function[print_shape](grid_dim=(100, 100, 100), block_dim=1)


# fn print_matrix_gpu[
#     r: Int, c: Int
# ](
#     ctx: DeviceContext, buff: DeviceBuffer, matrix: LayoutTensor[buff.type]
# ) raises:
#     alias rows = min(matrix.shape[0](), r)
#     alias cols = min(matrix.shape[1](), c)

#     host_buffer = enqueue_create_host_buf[matrix.dtype](ctx, rows * cols)
#     buff.enqueue_copy_to(host_buffer)
#     ctx.synchronize()

#     for i in range(rows * cols):
#         print(host_buffer[i], end="")
#         if i % cols == 0:
#             print()


fn dot[
    x: Int, y: Int, z: Int, dtype: DType
](
    ctx: DeviceContext,
    t1: LayoutTensor[dtype, Layout.row_major(x, y)],
    t2: LayoutTensor[dtype, Layout.row_major(y, z)],
) raises -> (
    DeviceBuffer[dtype],
    LayoutTensor[dtype, Layout.row_major(x, z), MutableAnyOrigin],
):
    # Assume that t1.cols is the largest
    # x is the rows for the weights -> 10
    # y is the rows for the train data -> 784
    # z is the cols for the train data, and the largest -> 42000
    tob, to = enqueue_create_matrix[rows=x, cols=z, dtype=dtype](ctx)

    fn dot_gpu(
        t1: __type_of(t1),
        t2: __type_of(t2),
        to: __type_of(to),
    ):
        t1x, t1y, t2y = block_idx.x, thread_idx.x, block_idx.y
        constrained[
            t1.shape[1]() == t2.shape[0](),
            "Dims does not match between t1 and t2.",
        ]()
        to[t1x, t2y] = t1[t1x, t1y] * t2[t1y, t2y]

    ctx.enqueue_function[dot_gpu](t1, t2, to, grid_dim=(x, z), block_dim=y)

    return tob, to


fn add[
    x: Int, y: Int, dtype: DType
](
    ctx: DeviceContext,
    t1: LayoutTensor[dtype, Layout.row_major(x, y)],
    t2: LayoutTensor[dtype, Layout.row_major(x, y)],
) raises -> (
    DeviceBuffer[dtype],
    LayoutTensor[dtype, Layout.row_major(x, y), MutableAnyOrigin],
):
    # Assume that t1.cols is the largest
    tob, to = enqueue_create_matrix[rows=x, cols=y, dtype=dtype](ctx)

    fn add_gpu(
        t1: __type_of(t1),
        t2: __type_of(t2),
        to: __type_of(to),
    ):
        t1x, t1y = thread_idx.x, block_idx.x
        constrained[
            t1.shape[0]() == t2.shape[0]() and t1.shape[1]() == t2.shape[1](),
            "Dims does not match between t1 and t2.",
        ]()
        to[t1x, t1y] = t1[t1x, t1y] * t2[t1x, t1y]

    ctx.enqueue_function[add_gpu](t1, t2, to, grid_dim=y, block_dim=x)

    return tob, to


fn add[
    x: Int, y: Int, dtype: DType
](
    ctx: DeviceContext,
    t1: LayoutTensor[dtype, Layout.row_major(x, y)],
    t2: LayoutTensor[dtype, Layout.row_major(x, 1)],
) raises -> (
    DeviceBuffer[dtype],
    LayoutTensor[dtype, Layout.row_major(x, y), MutableAnyOrigin],
):
    # Assume that t1.cols is the largest
    tob, to = enqueue_create_matrix[rows=x, cols=y, dtype=dtype](ctx)

    fn add_gpu(
        t1: __type_of(t1),
        t2: __type_of(t2),
        to: __type_of(to),
    ):
        t1x, t1y = thread_idx.x, block_idx.x
        constrained[
            t1.shape[0]() == t2.shape[0](),
            "Dims does not match between t1 and t2.",
        ]()
        to[t1x, t1y] = t1[t1x, t1y] * t2[t1x, 1]

    ctx.enqueue_function[add_gpu](t1, t2, to, grid_dim=y, block_dim=x)

    return tob, to


fn dot_add[
    x: Int, y: Int, z: Int, dtype: DType
](
    ctx: DeviceContext,
    t1: LayoutTensor[dtype, Layout.row_major(x, y)],
    t2: LayoutTensor[dtype, Layout.row_major(y, z)],
    t3: LayoutTensor[dtype, Layout.row_major(x, z)],
) raises -> (
    DeviceBuffer[dtype],
    LayoutTensor[dtype, Layout.row_major(x, z), MutableAnyOrigin],
):
    _db, dt = dot(ctx, t1, t2)
    return add(ctx, dt, t3)


fn dot_add[
    x: Int, y: Int, z: Int, dtype: DType
](
    ctx: DeviceContext,
    t1: LayoutTensor[dtype, Layout.row_major(x, y)],
    t2: LayoutTensor[dtype, Layout.row_major(y, z)],
    t3: LayoutTensor[dtype, Layout.row_major(x, 1)],
) raises -> (
    DeviceBuffer[dtype],
    LayoutTensor[dtype, Layout.row_major(x, z), MutableAnyOrigin],
):
    _db, dt = dot(ctx, t1, t2)
    return add(ctx, dt, t3)


fn relu[
    x: Int, y: Int, dtype: DType
](
    ctx: DeviceContext,
    ti: LayoutTensor[dtype, Layout.row_major(x, y), MutableAnyOrigin],
) raises -> (DeviceBuffer[dtype], __type_of(ti)):
    tob, to = enqueue_create_matrix[rows=x, cols=y, dtype = ti.dtype](ctx)

    fn relu_gpu(ti: __type_of(ti), to: __type_of(to)):
        xi, yi = thread_idx.x, block_idx.x
        to[xi, yi] = max(ti[xi, yi], 0)

    ctx.enqueue_function[relu_gpu](ti, to, grid_dim=y, block_dim=x)

    return tob, to


fn _sum[
    rows: Int, cols: Int, dtype: DType
](
    ctx: DeviceContext,
    ti: LayoutTensor[dtype, Layout.row_major(rows, cols), MutableAnyOrigin],
) raises -> Scalar[dtype]:
    alias simd_chunks = 2**13
    alias load_size = next_power_of_two(rows)

    _out_buff, out_matrix = enqueue_create_matrix[1, cols, dtype=dtype](ctx)

    fn _sum_gpu(
        ti: __type_of(ti), out: __type_of(out_matrix), mut sum: Scalar[dtype]
    ):
        tr = block_idx.x  # These are cols in the original tensor
        col_values = ti.load[load_size](tr, 0).shift_left[load_size - rows]()
        out[0, tr] = col_values.reduce_add()

        # barrier()

        # if block_idx.x == 0:
        #     iterations = 1 + cols // simd_chunks

        #     for i in range(iterations):
        #         rem = cols - simd_chunks * (i + 1)
        #         print("Last reduce at idx", i, "with rem", rem)

        #         accum = out.load[simd_chunks](0, simd_chunks * i)

        #         if rem < 0:
        #             sum += accum.shift_left[
        #                 simd_chunks - (cols % simd_chunks)
        #             ]().reduce_add()
        #             break

        #         sum += accum.reduce_add()

    sum = Scalar[dtype]()
    ctx.enqueue_function[_sum_gpu](
        ti.transpose(), out_matrix, sum, grid_dim=cols, block_dim=1
    )
    ctx.synchronize()

    return sum
    # total = Scalar[dtype]()
    # alias sum_times = 1 + cols // simd_chunks

    # @parameter
    # for i in range(sum_times):  # change it to be aware of 2^15 limitation
    #     alias rem = cols - simd_chunks * (i + 1)

    #     accum = host_tensor.load[simd_chunks](1, simd_chunks * i)

    #     @parameter
    #     if rem < 0:
    #         total += accum.shift_left[abs(rem)]().reduce_add()
    #         break

    #     total += accum.reduce_add()

    # return total


fn softmax[
    x: Int, y: Int, dtype: DType
](
    ctx: DeviceContext,
    ti: LayoutTensor[dtype, Layout.row_major(x, y), MutableAnyOrigin],
) raises -> (DeviceBuffer[dtype], __type_of(ti)):
    # alias rows = ti.shape[0]()
    # alias cols = ti.shape[1]()
    tob, to = enqueue_create_matrix[rows=x, cols=y, dtype = ti.dtype](ctx)

    fn softmax_gpu(
        ti: __type_of(ti), tensor_sum: Scalar[dtype], to: __type_of(to)
    ):
        xi, yi = block_idx.x, thread_idx.x

        to[xi, yi] = e ** (to[xi, yi] - tensor_sum)
        to[xi, yi] = to[xi, yi]

    tensor_sum = _sum(ctx, ti)
    # ctx.enqueue_function[softmax_gpu](
    #     ti, tensor_sum, to, grid_dim=x, block_dim=y
    # )

    return tob, to


fn forward_propagation[
    xr: Int, xc: Int, w1r: Int, w2r: Int, dtype: DType
](
    ctx: DeviceContext,
    w1: LayoutTensor[dtype, Layout.row_major(w1r, xr), MutableAnyOrigin],
    b1: LayoutTensor[dtype, Layout.row_major(w1r, xc), MutableAnyOrigin],
    w2: LayoutTensor[dtype, Layout.row_major(w2r, w1r), MutableAnyOrigin],
    b2: LayoutTensor[dtype, Layout.row_major(w2r, xc), MutableAnyOrigin],
    x: LayoutTensor[dtype, Layout.row_major(xr, xc)],
) raises -> (__type_of(b1), __type_of(b1), __type_of(b2), __type_of(b2)):
    # create an out tensor to hold the result.
    _z1b, z1 = dot_add(ctx, w1, x, b1)
    _a1b, a1 = relu(ctx, z1)
    _z2b, z2 = dot_add(ctx, w2, a1, b2)
    _a2b, a2 = softmax(ctx, z2)
    return z1, a1, z2, a2


fn forward_propagation[
    xr: Int, xc: Int, w1r: Int, w2r: Int, dtype: DType
](
    ctx: DeviceContext,
    w1: LayoutTensor[dtype, Layout.row_major(w1r, xr), MutableAnyOrigin],
    b1: LayoutTensor[dtype, Layout.row_major(w1r, 1), MutableAnyOrigin],
    w2: LayoutTensor[dtype, Layout.row_major(w2r, w1r), MutableAnyOrigin],
    b2: LayoutTensor[dtype, Layout.row_major(w2r, 1), MutableAnyOrigin],
    x: LayoutTensor[dtype, Layout.row_major(xr, xc)],
) raises -> (
    LayoutTensor[dtype, Layout.row_major(w1r, xc), MutableAnyOrigin],
    LayoutTensor[dtype, Layout.row_major(w1r, xc), MutableAnyOrigin],
    LayoutTensor[dtype, Layout.row_major(w2r, xc), MutableAnyOrigin],
    LayoutTensor[dtype, Layout.row_major(w2r, xc), MutableAnyOrigin],
):
    # create an out tensor to hold the result.
    _z1b, z1 = dot_add(ctx, w1, x, b1)
    _a1b, a1 = relu(ctx, z1)
    _z2b, z2 = dot_add(ctx, w2, a1, b2)
    _a2b, a2 = softmax(ctx, z2)
    return z1, a1, z2, a2
