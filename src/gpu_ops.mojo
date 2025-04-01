from gpu.host import DeviceContext, DeviceBuffer
from gpu import warp, barrier, thread_idx, block_idx
from gpu.memory import AddressSpace

from layout import Layout, LayoutTensor, composition, IntTuple
from bit import next_power_of_two
from math import ceil
from memory import stack_allocation

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

    print("[", end="")
    for i in range(rows * cols):
        print(host_buffer[i], end=",")
        if i % cols == 0:
            print("]", end="\n[")
    print("]")


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

    alias warps = y // 32 + (1 if y % 32 > 0 else 0)

    fn dot_gpu(
        t1: __type_of(t1),
        t2: __type_of(t2),
        to: __type_of(to),
    ):
        shared = stack_allocation[
            warps, dtype, address_space = AddressSpace.SHARED
        ]()

        t1x, t1y, t2y = block_idx.x, thread_idx.x, block_idx.y
        constrained[
            t1.shape[1]() == t2.shape[0](),
            "Dims does not match between t1 and t2.",
        ]()
        mulval = t1[t1x, t1y] * t2[t1y, t2y]
        shared[t1y // 32] = warp.sum(mulval)[0]

        barrier()

        if t1y == 0:
            to[t1x, t2y] = shared.load[
                width = next_power_of_two(warps)
            ]().reduce_add()

    ctx.enqueue_function[dot_gpu](t1, t2, to, grid_dim=(x, z), block_dim=y)

    return tob, to


# fn add[
#     x: Int, y: Int, dtype: DType
# ](
#     ctx: DeviceContext,
#     t1: LayoutTensor[dtype, Layout.row_major(x, y)],
#     t2: LayoutTensor[dtype, Layout.row_major(x, y)],
# ) raises -> (
#     DeviceBuffer[dtype],
#     LayoutTensor[dtype, Layout.row_major(x, y), MutableAnyOrigin],
# ):
#     # Assume that t1.cols is the largest
#     tob, to = enqueue_create_matrix[rows=x, cols=y, dtype=dtype](ctx)

#     fn add_gpu(
#         t1: __type_of(t1),
#         t2: __type_of(t2),
#         to: __type_of(to),
#     ):
#         t1x, t1y = thread_idx.x, block_idx.x
#         constrained[
#             t1.shape[0]() == t2.shape[0]() and t1.shape[1]() == t2.shape[1](),
#             "Dims does not match between t1 and t2.",
#         ]()
#         to[t1x, t1y] = t1[t1x, t1y] * t2[t1x, t1y]

#     ctx.enqueue_function[add_gpu](t1, t2, to, grid_dim=y, block_dim=x)

#     return tob, to


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
        to[t1x, t1y] = t1[t1x, t1y] * t2[t1x, 0]

    ctx.enqueue_function[add_gpu](t1, t2, to, grid_dim=y, block_dim=x)

    return tob, to


# fn dot_add[
#     x: Int, y: Int, z: Int, dtype: DType
# ](
#     ctx: DeviceContext,
#     t1: LayoutTensor[dtype, Layout.row_major(x, y)],
#     t2: LayoutTensor[dtype, Layout.row_major(y, z)],
#     t3: LayoutTensor[dtype, Layout.row_major(x, z)],
# ) raises -> (
#     DeviceBuffer[dtype],
#     LayoutTensor[dtype, Layout.row_major(x, z), MutableAnyOrigin],
# ):
#     _db, dt = dot(ctx, t1, t2)
#     return add(ctx, dt, t3)


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


fn sum_zero_axis[
    rows: Int, cols: Int, dtype: DType
](
    ctx: DeviceContext,
    ti: LayoutTensor[dtype, Layout.row_major(rows, cols), MutableAnyOrigin],
) raises -> (
    DeviceBuffer[dtype],
    LayoutTensor[dtype, Layout.row_major(cols), MutableAnyOrigin],
):
    alias warps = rows // 32 + (1 if rows % 32 > 0 else 0)
    out_buff, out_matrix = enqueue_create_matrix[cols, dtype=dtype](ctx)

    fn sum_zero_axis_gpu(
        tensor: __type_of(ti),
        out: __type_of(out_matrix),
    ):
        shared = stack_allocation[
            warps, Scalar[dtype], address_space = AddressSpace.SHARED
        ]()

        r, c = thread_idx.x, block_idx.x
        th_value = tensor.load[1](r, c)
        value = warp.sum(th_value)
        shared[r // 32] = value

        barrier()

        if thread_idx.x == 0:
            out[c] = shared.load[
                width = next_power_of_two(warps)
            ]().reduce_add()

    ctx.enqueue_function[sum_zero_axis_gpu](
        ti, out_matrix, grid_dim=cols, block_dim=1
    )

    return out_buff, out_matrix


fn softmax[
    rows: Int, cols: Int, dtype: DType
](
    ctx: DeviceContext,
    tib: DeviceBuffer[dtype],
    ti: LayoutTensor[dtype, Layout.row_major(rows, cols), MutableAnyOrigin],
) raises -> (DeviceBuffer[dtype], __type_of(ti)):
    tob, to = enqueue_create_matrix[rows=rows, cols=cols, dtype = ti.dtype](ctx)

    # CALC THE MAX VALUE IN ALL THE BUFFER
    max_v = Scalar[dtype]()

    host = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    tib.enqueue_copy_to(host)
    t = LayoutTensor[dtype, Layout.row_major(rows, cols)](host)

    ctx.synchronize()

    # TODO: Improve Performance
    for r in range(rows):
        for c in range(cols):
            max_v = max(t[r, c].reduce_max(), max_v)

    # reduce_1 = t.reshape[Layout.row_major(1024, warps)]()

    fn all_max(t: __type_of(ti)):
        shared = stack_allocation[
            cols // 32 + 1, dtype, address_space = AddressSpace.SHARED
        ]()

        r, c = thread_idx.x, block_idx.x
        tvalue = t.load[1](r, c)
        shared[r // 32] = warp.max(tvalue)

        barrier()

        if thread_idx.x == 0:
            max = shared.load[width = cols // 32 + 1]().reduce_max()
            print(max, end=" ")

    ctx.enqueue_function[all_max](ti, grid_dim=cols, block_dim=rows)
    print(max_v)

    # Do the exponential calculation
    fn _exp(ti: __type_of(to), max: Scalar[dtype], to: __type_of(to)):
        c, r = block_idx.x, thread_idx.x
        to[r, c] = e ** (ti[r, c] - max)

    ctx.enqueue_function[_exp](ti, max_v, to, grid_dim=cols, block_dim=rows)

    # Calculate the sum for each column. # TODO: Test it, since it's using load.
    _max_buff, sum_t = sum_zero_axis(ctx, to)

    # Divide the exp by the sum
    fn _div(to: __type_of(to), sum_t: __type_of(sum_t)):
        r, c = thread_idx.x, block_idx.x
        to[r, c] /= sum_t[c]

    ctx.enqueue_function[_div](to, sum_t, grid_dim=cols, block_dim=rows)

    return tob, to


# fn forward_propagation[
#     xr: Int, xc: Int, w1r: Int, w2r: Int, dtype: DType
# ](
#     ctx: DeviceContext,
#     w1: LayoutTensor[dtype, Layout.row_major(w1r, xr), MutableAnyOrigin],
#     b1: LayoutTensor[dtype, Layout.row_major(w1r, xc), MutableAnyOrigin],
#     w2: LayoutTensor[dtype, Layout.row_major(w2r, w1r), MutableAnyOrigin],
#     b2: LayoutTensor[dtype, Layout.row_major(w2r, xc), MutableAnyOrigin],
#     x: LayoutTensor[dtype, Layout.row_major(xr, xc)],
# ) raises -> (__type_of(b1), __type_of(b1), __type_of(b2), __type_of(b2)):
#     # create an out tensor to hold the result.
#     _z1b, z1 = dot_add(ctx, w1, x, b1)
#     _a1b, a1 = relu(ctx, z1)
#     _z2b, z2 = dot_add(ctx, w2, a1, b2)
#     _a2b, a2 = softmax(ctx, z2)
#     return z1, a1, z2, a2


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
    (
        DeviceBuffer[dtype],
        LayoutTensor[dtype, Layout.row_major(w1r, xc), MutableAnyOrigin],
    ),
    (
        DeviceBuffer[dtype],
        LayoutTensor[dtype, Layout.row_major(w1r, xc), MutableAnyOrigin],
    ),
    (
        DeviceBuffer[dtype],
        LayoutTensor[dtype, Layout.row_major(w2r, xc), MutableAnyOrigin],
    ),
    (
        DeviceBuffer[dtype],
        LayoutTensor[dtype, Layout.row_major(w2r, xc), MutableAnyOrigin],
    ),
):
    # create an out tensor to hold the result.
    z1b, z1 = dot_add(ctx, w1, x, b1)
    a1b, a1 = relu(ctx, z1)
    z2b, z2 = dot_add(ctx, w2, a1, b2)
    a2b, a2 = softmax(ctx, z2b, z2)
    return (z1b, z1), (a1b, a1), (z2b, z2), (a2b, a2)
