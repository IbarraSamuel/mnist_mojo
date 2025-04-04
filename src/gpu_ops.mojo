from gpu.host import DeviceContext, DeviceBuffer
from gpu import warp, barrier, thread_idx, block_idx
from gpu.memory import AddressSpace

from layout import Layout as LY
from layout import LayoutTensor, composition, IntTuple
from bit import next_power_of_two
from math import ceil
from memory import stack_allocation

from gpu_mem import (
    enqueue_create_matrix,
    enqueue_create_host_buf,
    MAX_BLOCKS_1D,
    MAX_BLOCKS_2D,
    MAX_BLOCKS_3D,
    Layout,
)

from math import e


# fn print_matrix[
#     r: Int = -1, c: Int = -1
# ](
#     ctx: DeviceContext, buff: DeviceBuffer, matrix: LayoutTensor[buff.type]
# ) raises:
#     alias rows = r if r != -1 else matrix.shape[0]()
#     alias cols = c if c != -1 else matrix.shape[1]()

#     host_buffer = enqueue_create_host_buf[matrix.dtype](ctx, rows * cols)
#     buff.enqueue_copy_to(host_buffer)
#     ctx.synchronize()

#     print("[", end="")
#     for i in range(rows * cols):
#         print(host_buffer[i], end=",")
#         if i % cols == 0:
#             print("]", end="\n[")
#     print("]")


# fn matrix_max[
#     dtype: DType, rows: Int, cols: Int, //
# ](
#     ctx: DeviceContext,
#     ti: LayoutTensor[dtype, Layout.row_major(rows, cols)],
# ) raises -> LayoutTensor[dtype, Layout.row_major(1), MutableAnyOrigin]:
#     out = ctx.enqueue_create_buffer[dtype](1)
#     out_t = LayoutTensor[dtype, Layout.row_major(1)](out)

#     fn all_max(
#         rd: Int,
#         cd: Int,
#         t: __type_of(ti),
#         final: __type_of(out_t),
#     ):
#         shared = stack_allocation[
#             32, dtype, address_space = AddressSpace.SHARED
#         ]()

#         r, c = thread_idx.x, block_idx.x
#         idx = r * cd + c
#         # Calculate row and col based on index.
#         rr, rc = idx // cols, idx % cols

#         tvalue = t[rr, rc][0]
#         value = warp.max(tvalue)
#         shared[r // 32] = value

#         barrier()

#         if thread_idx.x == 0:
#             max = shared.load[width=32]().reduce_max()
#             lr, lc = c // cols, c % cols
#             t[lr, lc] = max

#         if thread_idx.x == 0 and block_idx.x == 0 and cd == 1:
#             final[0] = t[0, 0][0]

#     new_cols = cols
#     new_rows = rows

#     # While we have more than 1 columns, let's do row reduction, and
#     while new_cols != 1:
#         elems, new_rows = new_cols * new_rows, 1024
#         while elems % new_rows != 0:
#             new_rows -= 1
#         new_cols = elems // new_ros

#         ctx.enqueue_function[all_max](
#             new_rows,
#             new_cols,
#             ti,
#             out_t,
#             grid_dim=new_cols,
#             block_dim=new_rows,
#         )

#         new_rows = 1

#     # ctx.synchronize()
#     return out_t


fn matrix_reduce[
    dtype: DType, //,
    warp_op: fn[dt: DType, w: Int, //] (SIMD[dt, w]) -> SIMD[dt, w],
    simd_op: fn[d: DType, s: Int] (SIMD[d, s]) -> SIMD[d, 1],
](
    ctx: DeviceContext,
    ti: LayoutTensor[dtype],
) raises -> LayoutTensor[
    dtype, Layout(1), MutableAnyOrigin
]:
    alias rows = ti.shape[0]()
    alias cols = ti.shape[1]()

    out = ctx.enqueue_create_buffer[dtype](1)
    out_t = LayoutTensor[dtype, Layout(1)](out)

    fn all_max(
        rd: Int,
        cd: Int,
        t: __type_of(ti),
        final: __type_of(out_t),
    ):
        shared = stack_allocation[
            32, dtype, address_space = AddressSpace.SHARED
        ]()

        r, c = thread_idx.x, block_idx.x
        idx = r * cd + c
        # Calculate row and col based on index.
        rr, rc = idx // cols, idx % cols

        tvalue = t[rr, rc][0]
        value = warp_op(tvalue)
        shared[r // 32] = value

        barrier()

        if thread_idx.x == 0:
            max = simd_op(shared.load[width=32]())
            lr, lc = c // cols, c % cols
            t[lr, lc] = max

        if thread_idx.x == 0 and block_idx.x == 0 and cd == 1:
            final[0] = t[0, 0][0]

    new_cols = cols
    new_rows = rows

    # While we have more than 1 columns, let's do row reduction, and
    while new_cols != 1:
        elems, new_rows = new_cols * new_rows, 1024
        while elems % new_rows != 0:
            new_rows -= 1
        new_cols = elems // new_rows

        ctx.enqueue_function[all_max](
            new_rows,
            new_cols,
            ti,
            out_t,
            grid_dim=new_cols,
            block_dim=new_rows,
        )

        new_rows = 1

    # ctx.synchronize()
    return out_t


# fn dot[
#     x: Int, y: Int, z: Int, dtype: DType
# ](
#     ctx: DeviceContext,
#     t1: LayoutTensor[dtype, Layout.row_major(x, y)],
#     t2: LayoutTensor[dtype, Layout.row_major(y, z)],
# ) raises -> (
#     DeviceBuffer[dtype],
#     LayoutTensor[dtype, Layout.row_major(x, z), MutableAnyOrigin],
# ):
#     tob, to = enqueue_create_matrix[rows=x, cols=z, dtype=dtype](ctx)

#     alias warps = y // 32 + (1 if y % 32 > 0 else 0)

#     fn dot_gpu(
#         t1: __type_of(t1),
#         t2: __type_of(t2),
#         to: __type_of(to),
#     ):
#         shared = stack_allocation[
#             warps, dtype, address_space = AddressSpace.SHARED
#         ]()

#         t1x, t1y, t2y = block_idx.x, thread_idx.x, block_idx.y

#         mulval = t1[t1x, t1y] * t2[t1y, t2y]
#         shared[t1y // 32] = warp.sum(mulval)[0]

#         barrier()

#         if t1y == 0:
#             to[t1x, t2y] = shared.load[
#                 width = next_power_of_two(warps)
#             ]().reduce_add()

#     ctx.enqueue_function[dot_gpu](t1, t2, to, grid_dim=(x, z), block_dim=y)

#     return tob, to


# fn dot[
#     x: Int, y: Int, z: Int, dtype: DType
# ](
#     ctx: DeviceContext,
#     t1: LayoutTensor[dtype, Layout(x, y)],
#     t2: LayoutTensor[dtype, Layout(y, z)],
# ) raises -> LayoutTensor[dtype, Layout(x, z), MutableAnyOrigin]:
#     _tob, to = enqueue_create_matrix[
#         dtype=dtype,
#         layout = Layout(x, z),
#     ](ctx)

#     alias warps = y // 32 + (1 if y % 32 > 0 else 0)

#     fn dot_gpu(
#         t1: __type_of(t1),
#         t2: __type_of(t2),
#         to: __type_of(to),
#     ):
#         shared = stack_allocation[
#             warps, dtype, address_space = AddressSpace.SHARED
#         ]()

#         t1x, t1y, t2y = block_idx.x, thread_idx.x, block_idx.y

#         mulval = t1[t1x, t1y] * t2[t1y, t2y]
#         shared[t1y // 32] = warp.sum(mulval)[0]

#         barrier()

#         if t1y == 0:
#             to[t1x, t2y] = shared.load[
#                 width = next_power_of_two(warps)
#             ]().reduce_add()

#     ctx.enqueue_function[dot_gpu](t1, t2, to, grid_dim=(x, z), block_dim=y)

#     return to


fn dot[
    dtype: DType
](
    ctx: DeviceContext,
    t1: LayoutTensor[dtype],
    t2: LayoutTensor[dtype],
) raises -> LayoutTensor[
    dtype, Layout(t1.shape[0](), t2.shape[1]()), MutableAnyOrigin
]:
    alias x = t1.shape[0]()
    alias z = t2.shape[1]()
    alias y = t1.shape[1]()
    alias y2 = t2.shape[0]()
    constrained[y == y2, "Dims should match."]()
    _tob, to = enqueue_create_matrix[
        dtype=dtype,
        layout = Layout(t1.shape[0](), t2.shape[1]()),
    ](ctx)

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

        mulval = t1[t1x, t1y] * t2[t1y, t2y]
        shared[t1y // 32] = warp.sum(mulval)[0]

        barrier()

        if t1y == 0:
            to[t1x, t2y] = shared.load[
                width = next_power_of_two(warps)
            ]().reduce_add()

    ctx.enqueue_function[dot_gpu](t1, t2, to, grid_dim=(x, z), block_dim=y)

    return to


fn add[
    dtype: DType
](
    ctx: DeviceContext,
    t1: LayoutTensor[dtype],
    t2: LayoutTensor[dtype],
) raises -> LayoutTensor[t1.dtype, t1.layout, MutableAnyOrigin]:
    alias x = t1.shape[0]()
    alias y = t1.shape[1]()
    alias x2 = t2.shape[0]()
    alias dim1 = t2.shape[1]()

    constrained[x == x2, "dims should match"]()
    constrained[dim1 == 1, "dim should be 1"]()

    # Assume that t1.cols is the largest
    _tob, to = enqueue_create_matrix(ctx, like=t1)

    fn add_gpu(
        t1: __type_of(t1),
        t2: __type_of(t2),
        to: __type_of(to),
    ):
        xi, yi = thread_idx.x, block_idx.x
        to[xi, yi] = t1[xi, yi] + t2[xi, 0]

    ctx.enqueue_function[add_gpu](t1, t2, to, grid_dim=y, block_dim=x)

    return to


fn add[
    dtype: DType,
    layout: LY,
](
    ctx: DeviceContext,
    t1: LayoutTensor[dtype, layout],
    t2: LayoutTensor[dtype, layout],
) raises -> LayoutTensor[dtype, layout, MutableAnyOrigin]:
    alias rows = t1.shape[0]()
    alias cols = t1.shape[1]()
    _, out = enqueue_create_matrix(ctx, like=t1)

    fn add_gpu(a: __type_of(t1), b: __type_of(t2), out: __type_of(out)):
        r, c = thread_idx.x, block_idx.x
        out[r, c] = a[r, c] + b[r, c]

    ctx.enqueue_function[add_gpu](t1, t2, out, grid_dim=cols, block_dim=rows)

    return out


fn sub[
    dtype: DType
](
    ctx: DeviceContext,
    t1: LayoutTensor[dtype],
    t2: LayoutTensor[dtype],
) raises -> LayoutTensor[t1.dtype, t1.layout, MutableAnyOrigin]:
    alias x = t1.shape[0]()
    alias y = t1.shape[1]()
    alias x2 = t2.shape[0]()
    alias dim1 = t2.shape[1]()

    constrained[x == x2, "dims should match"]()
    constrained[dim1 == 1, "dim should be 1"]()

    # Assume that t1.cols is the largest
    _tob, to = enqueue_create_matrix(ctx, like=t1)

    fn sub_gpu(
        t1: __type_of(t1),
        t2: __type_of(t2),
        to: __type_of(to),
    ):
        xi, yi = thread_idx.x, block_idx.x
        to[xi, yi] = t1[xi, yi] - t2[xi, 0]

    ctx.enqueue_function[sub_gpu](t1, t2, to, grid_dim=y, block_dim=x)

    return to


fn sub[
    dtype: DType,
    layout: LY,
](
    ctx: DeviceContext,
    t1: LayoutTensor[dtype, layout],
    t2: LayoutTensor[dtype, layout],
) raises -> LayoutTensor[dtype, layout, MutableAnyOrigin]:
    alias rows = t1.shape[0]()
    alias cols = t1.shape[1]()
    _, out = enqueue_create_matrix(ctx, like=t1)

    fn sub_gpu(a: __type_of(t1), b: __type_of(t2), out: __type_of(out)):
        r, c = thread_idx.x, block_idx.x
        out[r, c] = a[r, c] - b[r, c]

    ctx.enqueue_function[sub_gpu](t1, t2, out, grid_dim=cols, block_dim=rows)

    return out


fn mul[
    dtype: DType,
    layout: LY,
](
    ctx: DeviceContext,
    t1: LayoutTensor[dtype, layout],
    t2: LayoutTensor[dtype, layout],
) raises -> LayoutTensor[dtype, layout, MutableAnyOrigin]:
    alias rows = t1.shape[0]()
    alias cols = t1.shape[1]()
    _, out = enqueue_create_matrix(ctx, like=t1)

    fn mul_gpu(a: __type_of(t1), b: __type_of(t2), out: __type_of(out)):
        r, c = thread_idx.x, block_idx.x
        out[r, c] = a[r, c] * b[r, c]

    ctx.enqueue_function[mul_gpu](t1, t2, out, grid_dim=cols, block_dim=rows)

    return out


fn mul[
    dtype: DType,
    layout: LY,
](
    ctx: DeviceContext,
    t1: LayoutTensor[dtype, layout],
    t2: Scalar[dtype],
) raises -> LayoutTensor[dtype, layout, MutableAnyOrigin]:
    alias rows = t1.shape[0]()
    alias cols = t1.shape[1]()
    _, out = enqueue_create_matrix(ctx, like=t1)

    fn mul_gpu(a: __type_of(t1), b: __type_of(t2), out: __type_of(out)):
        r, c = thread_idx.x, block_idx.x
        out[r, c] = a[r, c] * b

    ctx.enqueue_function[mul_gpu](t1, t2, out, grid_dim=cols, block_dim=rows)

    return out


fn div[
    dtype: DType,
    layout: LY,
](
    ctx: DeviceContext,
    t1: LayoutTensor[dtype, layout],
    t2: LayoutTensor[dtype, layout],
) raises -> LayoutTensor[dtype, layout, MutableAnyOrigin]:
    alias rows = t1.shape[0]()
    alias cols = t1.shape[1]()
    _, out = enqueue_create_matrix(ctx, like=t1)

    fn div_gpu(a: __type_of(t1), b: __type_of(t2), out: __type_of(out)):
        r, c = thread_idx.x, block_idx.x
        out[r, c] = a[r, c] / b[r, c]

    ctx.enqueue_function[div_gpu](t1, t2, out, grid_dim=cols, block_dim=rows)

    return out


#     x: Int, y: Int, z: Int, dtype: DType
# ](
#     ctx: DeviceContext,
#     t1: LayoutTensor[dtype, Layout.row_major(x, y)],
#     t2: LayoutTensor[dtype, Layout.row_major(y, z)],
#     t3: LayoutTensor[dtype, Layout.row_major(x, 1)],
# ) raises -> (
#     DeviceBuffer[dtype],
#     LayoutTensor[dtype, Layout.row_major(x, z), MutableAnyOrigin],
# ):
#     _db, dt = dot[x, y, z, dtype](ctx, t1, t2)
#     return add[x, z, dtype](ctx, dt, t3)


# fn dot_add[
#     dtype: DType
# ](
#     ctx: DeviceContext,
#     t1: LayoutTensor[dtype],
#     t2: LayoutTensor[dtype],
#     t3: LayoutTensor[dtype],
# ) raises -> LayoutTensor[
#     dtype, Layout(t1.shape[0](), t2.shape[1]()), MutableAnyOrigin
# ]:
#     alias r1 = t1.shape[0]()
#     alias c2 = t2.shape[1]()
#     alias r3 = t3.shape[0]()
#     constrained[r1 == r3, "Dims should match"]()

#     dt = dot(ctx, t1, t2)
#     return add(ctx, dt, t3)


fn relu[
    dtype: DType
](
    ctx: DeviceContext,
    ti: LayoutTensor[dtype],
) raises -> LayoutTensor[
    dtype, ti.layout, MutableAnyOrigin
]:
    alias x = ti.shape[0]()
    alias y = ti.shape[1]()

    _tob, to = enqueue_create_matrix(ctx, like=ti)

    fn relu_gpu(ti: __type_of(ti), to: __type_of(to)):
        xi, yi = thread_idx.x, block_idx.x
        to[xi, yi] = max(ti[xi, yi], 0)

    ctx.enqueue_function[relu_gpu](ti, to, grid_dim=y, block_dim=x)

    return to


# fn sum_zero_axis[
#     rows: Int, cols: Int, dtype: DType
# ](
#     ctx: DeviceContext,
#     ti: LayoutTensor[dtype, Layout.row_major(rows, cols), MutableAnyOrigin],
# ) raises -> (
#     DeviceBuffer[dtype],
#     LayoutTensor[dtype, Layout.row_major(cols), MutableAnyOrigin],
# ):
#     alias warps = rows // 32 + (1 if rows % 32 > 0 else 0)
#     out_buff, out_matrix = enqueue_create_matrix[cols, dtype=dtype](ctx)

#     fn sum_zero_axis_gpu(
#         tensor: __type_of(ti),
#         out: __type_of(out_matrix),
#     ):
#         shared = stack_allocation[
#             warps, Scalar[dtype], address_space = AddressSpace.SHARED
#         ]()

#         r, c = thread_idx.x, block_idx.x
#         th_value = tensor.load[1](r, c)
#         value = warp.sum(th_value)
#         shared[r // 32] = value

#         barrier()

#         if thread_idx.x == 0:
#             out[c] = shared.load[
#                 width = next_power_of_two(warps)
#             ]().reduce_add()

#     ctx.enqueue_function[sum_zero_axis_gpu](
#         ti, out_matrix, grid_dim=cols, block_dim=1
#     )

#     return out_buff, out_matrix


fn sum_zero_axis[
    dtype: DType
](
    ctx: DeviceContext,
    ti: LayoutTensor[dtype],
) raises -> (
    DeviceBuffer[dtype],
    LayoutTensor[dtype, Layout(ti.shape[1]()), MutableAnyOrigin],
):
    alias rows = ti.shape[0]()
    alias cols = ti.shape[1]()
    alias warps = rows // 32 + (1 if rows % 32 > 0 else 0)
    out_buff, out_matrix = enqueue_create_matrix[size=cols, dtype=dtype](ctx)

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
    dtype: DType
](
    ctx: DeviceContext,
    ti: LayoutTensor[dtype],
) raises -> LayoutTensor[
    dtype, ti.layout, MutableAnyOrigin
]:
    alias rows = ti.shape[0]()
    alias cols = ti.shape[1]()

    _tob, to = enqueue_create_matrix(ctx, like=ti)

    # CALC THE MAX VALUE IN ALL THE BUFFER
    max_v = matrix_reduce[warp.max, SIMD.reduce_max](ctx, ti)

    # Do the exponential calculation
    fn _exp(ti: __type_of(to), max: __type_of(max_v), to: __type_of(to)):
        c, r = block_idx.x, thread_idx.x
        to[r, c] = e ** (ti[r, c] - max[0])

    ctx.enqueue_function[_exp](ti, max_v, to, grid_dim=cols, block_dim=rows)

    # Calculate the sum for each column. # TODO: Test it, since it's using load.
    _max_buff, sum_t = sum_zero_axis(ctx, to)

    # Divide the exp by the sum
    fn _div(to: __type_of(to), sum_t: __type_of(sum_t)):
        r, c = thread_idx.x, block_idx.x
        to[r, c] /= sum_t[c]

    ctx.enqueue_function[_div](to, sum_t, grid_dim=cols, block_dim=rows)

    return to


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


# fn forward_propagation[
#     xr: Int, xc: Int, w1r: Int, w2r: Int, dtype: DType
# ](
#     ctx: DeviceContext,
#     w1: LayoutTensor[dtype, Layout.row_major(w1r, xr), MutableAnyOrigin],
#     b1: LayoutTensor[dtype, Layout.row_major(w1r, 1), MutableAnyOrigin],
#     w2: LayoutTensor[dtype, Layout.row_major(w2r, w1r), MutableAnyOrigin],
#     b2: LayoutTensor[dtype, Layout.row_major(w2r, 1), MutableAnyOrigin],
#     x: LayoutTensor[dtype, Layout.row_major(xr, xc)],
# ) raises -> (
#     (
#         DeviceBuffer[dtype],
#         LayoutTensor[dtype, Layout.row_major(w1r, xc), MutableAnyOrigin],
#     ),
#     (
#         DeviceBuffer[dtype],
#         LayoutTensor[dtype, Layout.row_major(w1r, xc), MutableAnyOrigin],
#     ),
#     (
#         DeviceBuffer[dtype],
#         LayoutTensor[dtype, Layout.row_major(w2r, xc), MutableAnyOrigin],
#     ),
#     (
#         DeviceBuffer[dtype],
#         LayoutTensor[dtype, Layout.row_major(w2r, xc), MutableAnyOrigin],
#     ),
# ):
#     # create an out tensor to hold the result.
#     z1b, z1 = dot_add[w1r, xr, xc, dtype](ctx, w1, x, b1)
#     a1b, a1 = relu(ctx, z1)
#     z2b, z2 = dot_add[w2r, w1r, xc, dtype](ctx, w2, a1, b2)
#     a2b, a2 = softmax(ctx, z2)
#     return (z1b, z1), (a1b, a1), (z2b, z2), (a2b, a2)


fn forward_propagation[
    r: Int, c: Int, a: Int, b: Int, dtype: DType
](
    ctx: DeviceContext,
    x: LayoutTensor[dtype, Layout(r, c), MutableAnyOrigin],
    w1: LayoutTensor[dtype, Layout(a, r), MutableAnyOrigin],
    b1: LayoutTensor[dtype, Layout(a, 1), MutableAnyOrigin],
    w2: LayoutTensor[dtype, Layout(b, a), MutableAnyOrigin],
    b2: LayoutTensor[dtype, Layout(b, 1), MutableAnyOrigin],
) raises -> (
    LayoutTensor[dtype, Layout(a, c), MutableAnyOrigin],
    LayoutTensor[dtype, Layout(a, c), MutableAnyOrigin],
    LayoutTensor[dtype, Layout(b, c), MutableAnyOrigin],
    LayoutTensor[dtype, Layout(b, c), MutableAnyOrigin],
):
    # The problem is in the dot product.
    alias D = LayoutTensor[dtype, Layout(a, c), MutableAnyOrigin]
    _d1 = rebind[D](dot(ctx, w1, x))  # Why it doesn't work automagically?
    z1 = add(ctx, _d1, b1)

    a1 = relu(ctx, z1)

    alias D2 = LayoutTensor[dtype, Layout(b, c), MutableAnyOrigin]
    _d2 = rebind[D2](dot(ctx, w2, a1))  # Why it doesn't work automagically?
    z2 = add(ctx, _d2, b2)

    a2 = softmax(ctx, z2)

    return z1, a1, z2, a2


fn forward_propagation[
    dtype: DType
](
    ctx: DeviceContext,
    x: LayoutTensor[dtype],
    w1: LayoutTensor[dtype],
    b1: LayoutTensor[dtype],
    w2: LayoutTensor[dtype],
    b2: LayoutTensor[dtype],
) raises -> (
    LayoutTensor[dtype, Layout(w1.shape[0](), x.shape[1]()), MutableAnyOrigin],
    LayoutTensor[dtype, Layout(w1.shape[0](), x.shape[1]()), MutableAnyOrigin],
    LayoutTensor[dtype, Layout(w2.shape[0](), x.shape[1]()), MutableAnyOrigin],
    LayoutTensor[dtype, Layout(w2.shape[0](), x.shape[1]()), MutableAnyOrigin],
):
    alias r: Int = x.shape[0]()
    alias c: Int = x.shape[1]()
    alias a: Int = w1.shape[0]()
    alias b: Int = w2.shape[0]()

    constrained[w1.shape[1]() == r]()
    constrained[b1.shape[0]() == a]()
    constrained[b1.shape[1]() == 1]()
    constrained[w2.shape[1]() == a]()
    constrained[b2.shape[0]() == b]()
    constrained[b2.shape[1]() == 1]()

    # The problem is in the dot product.
    _d1 = dot(ctx, w1, x)

    z1 = add(ctx, _d1, b1)

    a1 = relu(ctx, z1)

    _d2 = rebind[LayoutTensor[dtype, Layout(b, c), MutableAnyOrigin]](
        dot(ctx, w2, a1)
    )
    z2 = add(ctx, _d2, b2)

    a2 = softmax(ctx, z2)

    return z1, a1, z2, a2


# fn forward_propagation[
#     dtype: DType, //
# ](
#     ctx: DeviceContext,
#     x: LayoutTensor[dtype, Layout(r, c), MutableAnyOrigin],
#     w1: LayoutTensor[dtype, Layout(a, r), MutableAnyOrigin],
#     b1: LayoutTensor[dtype, Layout(a, 1), MutableAnyOrigin],
#     w2: LayoutTensor[dtype, Layout(b, a), MutableAnyOrigin],
#     b2: LayoutTensor[dtype, Layout(b, 1), MutableAnyOrigin],
# ) raises -> (
#     LayoutTensor[dtype, Layout(a, c), MutableAnyOrigin],
#     LayoutTensor[dtype, Layout(a, c), MutableAnyOrigin],
#     LayoutTensor[dtype, Layout(b, c), MutableAnyOrigin],
#     LayoutTensor[dtype, Layout(b, c), MutableAnyOrigin],
# ):
#     # The problem is in the dot product.
#     _d1 = dot[a, r, c](ctx, w1, x)
#     z1 = add(ctx, _d1, b1)

#     a1 = relu(ctx, z1)

#     _d2 = dot[b, a, c](ctx, w2, a1)
#     z2 = add(ctx, _d2, b2)

#     a2 = softmax(ctx, z2)

#     return z1, a1, z2, a2


fn der_relu[
    dtype: DType
](ctx: DeviceContext, t: LayoutTensor[dtype]) raises -> LayoutTensor[
    dtype, t.layout, MutableAnyOrigin
]:
    alias rows = t.shape[0]()
    alias cols = t.shape[1]()
    _, out = enqueue_create_matrix(ctx, like=t)

    fn is_positive(t: __type_of(t), o: __type_of(out)):
        r, c = thread_idx.x, block_idx.x
        o[r, c] = (t[r, c] > 0).cast[dtype]()

    ctx.enqueue_function[is_positive](t, out, grid_dim=cols, block_dim=rows)


fn one_hot_y[
    dtype: DType, y_len: Int, max_y: Int
](
    ctx: DeviceContext, t: LayoutTensor[dtype, Layout(y_len)]
) raises -> LayoutTensor[dtype, Layout(max_y + 1, y_len), MutableAnyOrigin]:
    alias layout = Layout(y_len, max_y + 1)
    yb, y = enqueue_create_matrix[layout=layout, dtype=dtype](ctx)
    yb = yb.enqueue_fill(0)

    # Need to do the one hot thing.

    yt = rebind[
        LayoutTensor[dtype, Layout(max_y + 1, y_len), MutableAnyOrigin]
    ](y.transpose())
    return yt


fn backward_propagation[
    xr: Int, xc: Int, w1r: Int, w2r: Int, dtype: DType
](
    ctx: DeviceContext,
    x: LayoutTensor[dtype, Layout(xr, xc)],
    z1: LayoutTensor[dtype, Layout(w1r, xc)],
    a1: LayoutTensor[dtype, z1.layout],
    # z2: LayoutTensor[dtype, Layout.row_major(w2r, xc)],
    a2: LayoutTensor[dtype, Layout(w2r, xc)],
    # w1: LayoutTensor[dtype, Layout.row_major(w1r, xr)],
    w2: LayoutTensor[dtype, Layout(w2r, w1r)],
    # y: LayoutTensor[dtype, Layout.row_major(xr, xc)],
    one_hot_y: LayoutTensor[dtype, a2.layout],
) raises -> (
    LayoutTensor[dtype, Layout(w1r, xr), MutableAnyOrigin],
    LayoutTensor[dtype, Layout(1), MutableAnyOrigin],
    LayoutTensor[dtype, Layout(w2r, w1r), MutableAnyOrigin],
    LayoutTensor[dtype, Layout(1), MutableAnyOrigin],
):
    alias m = x.shape[1]()
    alias mi = (1 / m).cast[dtype]()

    dz2 = sub(ctx, a2, one_hot_y)
    alias _DW2 = LayoutTensor[dtype, Layout(w2r, w1r), MutableAnyOrigin]
    _dw2 = dot(ctx, sub(ctx, a2, one_hot_y), a1.transpose())
    dw2 = mul(ctx, rebind[_DW2](_dw2), mi)

    sum_dz2 = matrix_reduce[warp_op = warp.sum, simd_op = SIMD.reduce_add](
        ctx, dz2
    )
    db2 = mul(ctx, sum_dz2, mi)
    alias _DZ1 = LayoutTensor[dtype, Layout(w1r, xc), MutableAnyOrigin]
    _dz1 = dot(ctx, w2.transpose(), dz2)
    drelu = der_relu(ctx, z1)
    dz1 = mul(ctx, rebind[_DZ1](_dz1), drelu)

    alias _DW1 = LayoutTensor[dtype, Layout(w1r, xr), MutableAnyOrigin]
    _dw1 = dot(ctx, dz1, x.transpose())
    dw1 = mul(ctx, rebind[_DW1](_dw1), mi)

    max_dz1 = matrix_reduce[warp_op = warp.sum, simd_op = SIMD.reduce_add](
        ctx, dz1
    )
    db1 = mul(ctx, max_dz1, mi)
    return dw1, db1, dw2, db2


# dz2 = a2 - one_hot_y
# dw2 = 1 / m * dz2.dot(dz2)
