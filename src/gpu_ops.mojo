from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, composition, IntTuple
from gpu.id import thread_idx, block_idx


from gpu_mem import enqueue_create_matrix, enqueue_create_host_buf

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
    tob, to = enqueue_create_matrix[rows=x, cols=z, dtype=dtype](ctx)

    fn dot_add_gpu(
        t1: __type_of(t1),
        t2: __type_of(t2),
        t3: __type_of(t3),
        to: __type_of(t3),
    ):
        t1x, t1y, t2y = thread_idx.x, thread_idx.y, thread_idx.z
        constrained[
            t1.shape[1]() == t2.shape[0](),
            "Dims does not match between t1 and t2.",
        ]()
        constrained[
            t1.shape[0]() == t3.shape[0]() and t2.shape[1]() == t3.shape[1](),
            "Dims does not match between t1, t2 and t3.",
        ]()
        to[t1x, t2y] = t1[t1x, t1y] * t2[t1y, t2y] + t3[t1x, t2y]

    ctx.enqueue_function[dot_add_gpu](
        t1, t2, t3, to, grid_dim=1, block_dim=(x, y, z)
    )

    return tob, to


fn relu_gpu(ti: LayoutTensor, to: __type_of(ti)):
    xi, yi = block_idx.x, block_idx.y
    to[xi, yi] = max(ti[xi, yi], 0)


fn relu[
    x: Int, y: Int, dtype: DType
](
    ctx: DeviceContext,
    ti: LayoutTensor[dtype, Layout.row_major(x, y), MutableAnyOrigin],
) raises -> (DeviceBuffer[dtype], __type_of(ti)):
    # alias rows = ti.shape[0]()
    # alias cols = ti.shape[1]()
    tob, to = enqueue_create_matrix[rows=x, cols=y, dtype = ti.dtype](ctx)

    ctx.enqueue_function[relu_gpu](ti, to, grid_dim=1, block_dim=(x, y))

    return tob, to


fn softmax_gpu(ti: LayoutTensor, to: __type_of(ti)):
    alias ti_x = ti.shape[0]()
    alias ti_y = ti.shape[1]()

    xi, yi = block_idx.x, block_idx.y
    tot = ti.load[ti_x * ti_y](0, 0).reduce_add()

    to[xi, yi] = ti[xi, yi] - tot
    to[xi, yi] = e ** to[xi, yi]
    to[xi, yi] = to[xi, yi]


fn softmax[
    x: Int, y: Int, dtype: DType
](
    ctx: DeviceContext,
    ti: LayoutTensor[dtype, Layout.row_major(x, y), MutableAnyOrigin],
) raises -> (DeviceBuffer[dtype], __type_of(ti)):
    # alias rows = ti.shape[0]()
    # alias cols = ti.shape[1]()
    tob, to = enqueue_create_matrix[rows=x, cols=y, dtype = ti.dtype](ctx)

    ctx.enqueue_function[softmax_gpu](ti, to, grid_dim=x, block_dim=(x, y))

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
    _, z1 = dot_add(ctx, w1, x, b1)
    _, a1 = relu(ctx, z1)
    _, z2 = dot_add(ctx, w2, a1, b2)
    _, a2 = softmax(ctx, z2)
    return z1, a1, z2, a2
