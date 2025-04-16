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


fn matrix_reduce[
    dtype: DType,
    ly: LY, //,
    warp_op: fn[dt: DType, w: Int, //] (SIMD[dt, w]) -> SIMD[dt, w],
    simd_op: fn[d: DType, s: Int] (SIMD[d, s]) -> SIMD[d, 1],
](
    ctx: DeviceContext,
    ti: LayoutTensor[dtype, ly],
) raises -> LayoutTensor[
    dtype, Layout(1), MutableAnyOrigin
]:
    alias rows = ti.shape[0]()
    alias cols = ti.shape[1]()

    _, out_t = enqueue_create_matrix[size=1, dtype=dtype](ctx)

    fn all_reduce(
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
            final[0, 0] = t[0, 0][0]

    new_cols = cols
    new_rows = rows

    # While we have more than 1 columns, let's do row reduction, and
    while new_cols != 1:
        elems, new_rows = new_cols * new_rows, 1024
        while elems % new_rows != 0:
            new_rows -= 1
        new_cols = elems // new_rows

        ctx.enqueue_function[all_reduce](
            new_rows,
            new_cols,
            ti,
            out_t,
            grid_dim=new_cols,
            block_dim=new_rows,
        )

        new_rows = 1

    return out_t


fn argmax_a0[
    dtype: DType, rows: Int, cols: Int
](
    ctx: DeviceContext, t: LayoutTensor[dtype, Layout(rows, cols)]
) raises -> LayoutTensor[dtype, Layout(cols), MutableAnyOrigin]:
    _, out = enqueue_create_matrix[Layout(cols), dtype](ctx)

    fn reduce_a0(t: __type_of(t), out: __type_of(out)):
        r, c = thread_idx.x, block_idx.x
        val = t[r, c]
        mx_val = warp.max(val)
        out[c] = r if val == mx_val else out[c]

    ctx.enqueue_function[reduce_a0](t, out, grid_dim=cols, block_dim=rows)
    return out


fn dot_large[
    dtype: DType, x_dim: Int, z_dim: Int, y_dim: Int
](
    ctx: DeviceContext,
    t1: LayoutTensor[dtype, Layout(x_dim, z_dim)],
    t2: LayoutTensor[dtype, Layout(z_dim, y_dim)],
) raises -> LayoutTensor[dtype, Layout(x_dim, y_dim), MutableAnyOrigin]:
    """
    When the common dimension is too large, we need to reduce it using other tecniques.
    I will try to use warps to iterate over each pixel and aggregate the result.

    With two matrices like:
    [a b c d e f]
    [g h i j k l]

    [a b c]
    [d e f]
    [g h i]
    [j k l]
    [m n o]
    [p q r]

    dims are (2, 6) and (6, 3)
    We expect a result of: 2, 3

    When the dimensions (a, b) (b, c)
    if a, b could fit in a block, we can use each block to calculate each `pixel`.
    Then, a shared `stack_allocation` per block can help us to do the total sum.
    Tricky part is, how can we store the intermediate results while iterating the dimension b?

    I'll not create a bigger tensor to get all things in, I will create the right shape for the final
    tensor. Then, for each pixel, iterate in batches of 1024, since this is the total a block can handle.
    Within each 1024 batch, I will use 32 warps of 32 threads, to resolve the assigned pixel.
    """

    buff = ctx.enqueue_create_buffer[dtype](x_dim * y_dim)
    out = LayoutTensor[dtype, Layout(x_dim, y_dim), MutableAnyOrigin](buff)

    alias block_dim: Int = min(1024, z_dim)
    alias repeat_threads = z_dim // 1024 + (1 if z_dim % 1024 > 0 else 0)
    """A single block can handle 1024 threads, if we treat each block as one pixel unit,
    Then we need to split the work for the specific pixel within the block.
    Each block has 1024 workers, so doing ceil(z_dim / 1024) should say how many times we should
    repeat the work to finalize the z_dim.
    If this number is one, then we should recalculate the block_dim
    """
    alias warps = 32  # warps that could be used in a single block execution.

    fn _calc_pixel(t1: __type_of(t1), t2: __type_of(t2), o: __type_of(out)):
        # Each time a warp finishes, we can store it's result here.
        shared = stack_allocation[
            warps, dtype, address_space = AddressSpace.SHARED
        ]()
        shared.store(0, SIMD[dtype, warps]())
        x, y, i = block_idx.x, block_idx.y, thread_idx.x

        z = Scalar[dtype]()

        for blk in range(repeat_threads):
            # Move the i index by blk * 1024, since other threads will handle
            # all other values.
            ii = blk * 1024 + i
            result_i = z if ii >= z_dim else t1[x, ii] * t2[ii, y]
            result = warp.sum(result_i)

            shared[i // warps] += result[0]  # Save in shared block memory

        barrier()

        if i == 0:  # In only one thread, all blocks so all pixels:
            o[x, y] = shared.load[width=32]().reduce_add()

    ctx.enqueue_function[_calc_pixel](
        t1, t2, out, grid_dim=(x_dim, y_dim), block_dim=block_dim
    )

    return out


fn dot[
    dtype: DType,
    x: Int,
    y: Int,
    z: Int,
](
    ctx: DeviceContext,
    t1: LayoutTensor[dtype, Layout(x, y)],
    t2: LayoutTensor[dtype, Layout(y, z)],
) raises -> LayoutTensor[dtype, Layout(x, z), MutableAnyOrigin]:
    """
    Calc the dot product.

    Assume that t2.cols is the largest.
    Then, t2.rows is the lowest
    Then, t1.cols fits in a single block.
    This is important since we can use warps to aggregate results
    for a single block.
    x -> 42000 ==> Largest -> Each one represents a block
    y -> 10 ==> Shortest -> each one could represent block or thread
    z -> 784 ==> Medium -> Could be block or thread, preffered to be a thread.
    Imagine:
    [1 1] [2 2 2] -> [3 3 3]
    [1 1] [2 2 2] -> [3 3 3]
    [1 1]         -> [3 3 3]
    [1 1]         -> [3 3 3]
    [1 1]         -> [3 3 3]
        We can calculate easily the multiplication, since we never depend on x or z
    We need to load y values from t1 and t2 (in a transpose manner or changing it to col_major), then multiply them
    to then, reduce add. And we can store it on the desired position

    like:
    (t1[xi, :] * t2[:, zi]).reduce_add()
    We can use a minimatrix with the results, to them collapse into a result as an option

    value = t1[xi, yi] * t2[yi, zi]
    and then, warp them, into another matrix using
    final[xi, zi] = warp(value)
    # We can always use the other tecnique to sum sub warps

    x is the rows for the weights -> 10
    y is the rows for the train data -> 784
    z is the cols for the train data, and the largest -> 42000
    """
    # alias x = t1.shape[0]()
    # alias z = t2.shape[1]()
    # alias y = t1.shape[1]()
    # alias y2 = t2.shape[0]()
    # constrained[y == y2, "Dims should match."]()
    # print(x, y, z)
    _tob, to = enqueue_create_matrix[dtype=dtype, layout = Layout(x, z)](ctx)

    # What happen if y > 1024?
    # We will need to collapse multiple blocks into a single value.
    if y > 1024:
        return dot_large(ctx, t1, t2)

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
            # constrained[dim1 == 1, "dim should be 1"]()
            to[t1x, t2y] = (
                shared.load[width = next_power_of_two(warps)]()
                .shift_left[next_power_of_two(warps) - warps]()
                .reduce_add()
            )

    ctx.enqueue_function[dot_gpu](t1, t2, to, grid_dim=(x, z), block_dim=y)

    return to


fn add[
    dtype: DType, a: Int, b: Int
](
    ctx: DeviceContext,
    t1: LayoutTensor[dtype, Layout(a, b)],
    t2: LayoutTensor[dtype, Layout(a)],
) raises -> LayoutTensor[t1.dtype, t1.layout, MutableAnyOrigin]:
    # Assume that t1.cols is the largest
    _tob, to = enqueue_create_matrix(ctx, like=t1)

    fn add_gpu(
        t1: __type_of(t1),
        t2: __type_of(t2),
        to: __type_of(to),
    ):
        xi, yi = thread_idx.x, block_idx.x
        to[xi, yi] = t1[xi, yi] + t2[xi]

    ctx.enqueue_function[add_gpu](t1, t2, to, grid_dim=b, block_dim=a)

    return to


fn add[
    dtype: DType, a: Int, b: Int
](
    ctx: DeviceContext,
    t1: LayoutTensor[dtype, Layout(a, b)],
    t2: LayoutTensor[dtype, Layout(a, b)],
) raises -> LayoutTensor[dtype, Layout(a, b), MutableAnyOrigin]:
    _, out = enqueue_create_matrix(ctx, like=t1)

    fn add_gpu(a: __type_of(t1), b: __type_of(t2), out: __type_of(out)):
        r, c = thread_idx.x, block_idx.x
        out[r, c] = a[r, c] + b[r, c]

    ctx.enqueue_function[add_gpu](t1, t2, out, grid_dim=b, block_dim=a)

    return out


fn sub[
    dtype: DType,
    a: Int,
    b: Int,
](
    ctx: DeviceContext,
    t1: LayoutTensor[dtype, Layout(a, b)],
    t2: LayoutTensor[dtype, Layout(a)],
) raises -> LayoutTensor[t1.dtype, t1.layout, MutableAnyOrigin]:
    alias x = t1.shape[0]()
    alias y = t1.shape[1]()
    alias x2 = t2.shape[0]()

    constrained[x == x2, "dims should match"]()

    # Assume that t1.cols is the largest
    _tob, to = enqueue_create_matrix(ctx, like=t1)

    fn sub_gpu(
        t1: __type_of(t1),
        t2: __type_of(t2),
        to: __type_of(to),
    ):
        xi, yi = thread_idx.x, block_idx.x
        to[xi, yi] = t1[xi, yi] - t2[xi]

    ctx.enqueue_function[sub_gpu](t1, t2, to, grid_dim=y, block_dim=x)

    return to


fn sub[
    dtype: DType, a: Int
](
    ctx: DeviceContext,
    t1: LayoutTensor[dtype, Layout(a)],
    k: LayoutTensor[dtype, Layout(1)],
) raises -> LayoutTensor[t1.dtype, t1.layout, MutableAnyOrigin]:
    _tob, to = enqueue_create_matrix(ctx, like=t1)

    fn sub_gpu(
        t1: __type_of(t1),
        k: __type_of(k),
        to: __type_of(to),
    ):
        xi = thread_idx.x
        to[xi] = t1[xi] - k[0]

    ctx.enqueue_function[sub_gpu](t1, k, to, grid_dim=1, block_dim=a)

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
](
    ctx: DeviceContext,
    t1: LayoutTensor[dtype, Layout(1)],
    t2: Scalar[dtype],
) raises -> LayoutTensor[dtype, Layout(1), MutableAnyOrigin]:
    alias size = t1.shape[0]()
    _, out = enqueue_create_matrix(ctx, like=t1)

    fn mul_gpu(a: __type_of(t1), b: __type_of(t2), out: __type_of(out)):
        i = thread_idx.x
        out[i] = a[i] * b

    ctx.enqueue_function[mul_gpu](t1, t2, out, grid_dim=1, block_dim=1)

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


fn relu[
    dtype: DType,
    ly1: LY,
](
    ctx: DeviceContext,
    ti: LayoutTensor[dtype, ly1],
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


fn sum_zero_axis[
    dtype: DType,
    layout: LY,
](
    ctx: DeviceContext,
    ti: LayoutTensor[dtype, layout],
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
    dtype: DType,
    ly: LY,
](
    ctx: DeviceContext,
    ti: LayoutTensor[dtype, ly],
) raises -> LayoutTensor[
    dtype, ly, MutableAnyOrigin
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


fn forward_propagation[
    r: Int, c: Int, a: Int, b: Int, dtype: DType
](
    ctx: DeviceContext,
    x: LayoutTensor[dtype, Layout(r, c), MutableAnyOrigin],
    w1: LayoutTensor[dtype, Layout(a, r), MutableAnyOrigin],
    b1: LayoutTensor[dtype, Layout(a), MutableAnyOrigin],
    w2: LayoutTensor[dtype, Layout(b, a), MutableAnyOrigin],
    b2: LayoutTensor[dtype, Layout(b), MutableAnyOrigin],
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


# fn forward_propagation[
#     dtype: DType
# ](
#     ctx: DeviceContext,
#     x: LayoutTensor[dtype],
#     w1: LayoutTensor[dtype],
#     b1: LayoutTensor[dtype],
#     w2: LayoutTensor[dtype],
#     b2: LayoutTensor[dtype],
# ) raises -> (
#     LayoutTensor[dtype, Layout(w1.shape[0](), x.shape[1]()), MutableAnyOrigin],
#     LayoutTensor[dtype, Layout(w1.shape[0](), x.shape[1]()), MutableAnyOrigin],
#     LayoutTensor[dtype, Layout(w2.shape[0](), x.shape[1]()), MutableAnyOrigin],
#     LayoutTensor[dtype, Layout(w2.shape[0](), x.shape[1]()), MutableAnyOrigin],
# ):
#     alias r: Int = x.shape[0]()
#     alias c: Int = x.shape[1]()
#     alias a: Int = w1.shape[0]()
#     alias b: Int = w2.shape[0]()

#     constrained[w1.shape[1]() == r]()
#     constrained[b1.shape[0]() == a]()
#     constrained[b1.shape[1]() == 1]()
#     constrained[w2.shape[1]() == a]()
#     constrained[b2.shape[0]() == b]()
#     constrained[b2.shape[1]() == 1]()

#     # The problem is in the dot product.
#     _d1 = dot(ctx, w1, x)

#     z1 = add(ctx, _d1, b1)

#     a1 = relu(ctx, z1)

#     _d2 = rebind[LayoutTensor[dtype, Layout(b, c), MutableAnyOrigin]](
#         dot(ctx, w2, a1)
#     )
#     z2 = add(ctx, _d2, b2)

#     a2 = softmax(ctx, z2)

#     return z1, a1, z2, a2


fn der_relu[
    dtype: DType,
    ly: LY,
](ctx: DeviceContext, t: LayoutTensor[dtype, ly]) raises -> LayoutTensor[
    dtype, t.layout, MutableAnyOrigin
]:
    alias rows = t.shape[0]()
    alias cols = t.shape[1]()
    _, out = enqueue_create_matrix(ctx, like=t)

    fn is_positive(t: __type_of(t), o: __type_of(out)):
        r, c = thread_idx.x, block_idx.x
        o[r, c] = (t[r, c] > 0).cast[dtype]()

    ctx.enqueue_function[is_positive](t, out, grid_dim=cols, block_dim=rows)

    return out


fn one_hot_y[
    dtype: DType, y_len: Int, max_y: Int
](
    ctx: DeviceContext, t: LayoutTensor[dtype, Layout(y_len)]
) raises -> LayoutTensor[dtype, Layout(max_y + 1, y_len), MutableAnyOrigin]:
    alias layout = Layout(y_len, max_y + 1)
    yb, y = enqueue_create_matrix[layout=layout, dtype=dtype](ctx)
    yb = yb.enqueue_fill(0)

    # Need to do the one hot thing.

    yt = rebind[LayoutTensor[dtype, Layout(max_y + 1, y_len), y.origin]](
        y.transpose()
    )
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
    alias m: Int = x.shape[1]()
    alias mi: Scalar[dtype] = (1 / m).cast[dtype]()

    # dw2
    dz2 = sub(ctx, a2, one_hot_y)
    # a2_sub = sub(ctx, a2, one_hot_y)
    alias A1T = LayoutTensor[a1.dtype, Layout(xc, w1r), a1.origin]
    a1t = rebind[A1T](a1.transpose())
    _dw2 = dot_large(ctx, dz2, a1t)
    dw2 = mul(ctx, _dw2, mi)

    # dz1
    sum_dz2 = matrix_reduce[warp.sum, SIMD.reduce_add](ctx, dz2)
    db2 = mul(ctx, sum_dz2, mi)
    alias W2T = LayoutTensor[a1.dtype, Layout(w1r, w2r), MutableAnyOrigin]
    w2t = rebind[W2T](w2.transpose())
    dz1 = dot(ctx, w2t, dz2)
    drelu = der_relu(ctx, z1)
    dz1 = mul(ctx, dz1, drelu)

    # # dw1
    alias XT = LayoutTensor[dtype, Layout(xc, xr), MutableAnyOrigin]
    # print(x.layout.shape)
    xt = rebind[XT](x.transpose())
    # print(dz1.layout.shape, xt.layout.shape)
    _dw1 = dot_large(ctx, dz1, xt)  # ILEGAL ADDRESS
    dw1 = mul(ctx, _dw1, mi)

    # # db1
    sum_dz1 = matrix_reduce[warp.sum, SIMD.reduce_add](ctx, dz1)
    db1 = mul(ctx, sum_dz1, mi)
    return dw1, db1, dw2, db2


fn update_parameters[
    dtype: DType, //, alpha: Scalar[dtype], w1l: LY, b11: Int, w2l: LY, b12: Int
](
    ctx: DeviceContext,
    w1: LayoutTensor[dtype, w1l],
    b1: LayoutTensor[dtype, Layout(b11)],
    w2: LayoutTensor[dtype, w2l],
    b2: LayoutTensor[dtype, Layout(b12)],
    dw1: LayoutTensor[dtype, w1l],
    db1: LayoutTensor[dtype, Layout(1)],
    dw2: LayoutTensor[dtype, w2l],
    db2: LayoutTensor[dtype, Layout(1)],
    # alpha: Scalar[dtype],
) raises -> (
    LayoutTensor[dtype, w1l, MutableAnyOrigin],
    LayoutTensor[dtype, Layout(b11), MutableAnyOrigin],
    LayoutTensor[dtype, w2l, MutableAnyOrigin],
    LayoutTensor[dtype, Layout(b12), MutableAnyOrigin],
):
    w1f = sub(ctx, w1, mul(ctx, dw1, alpha))
    b1f = sub(ctx, b1, mul(ctx, db1, alpha))
    w2f = sub(ctx, w2, mul(ctx, dw2, alpha))
    b2f = sub(ctx, b2, mul(ctx, db2, alpha))
    return w1f, b1f, w2f, b2f


fn get_predictions[
    dtype: DType, r: Int, c: Int
](
    ctx: DeviceContext, t: LayoutTensor[dtype, Layout(r, c)]
) raises -> LayoutTensor[dtype, Layout(c), MutableAnyOrigin]:
    return argmax_a0(ctx, t)


fn print_accuracy[
    dtype: DType, size: Int
](
    ctx: DeviceContext,
    t1: LayoutTensor[dtype, Layout(size)],
    t2: LayoutTensor[dtype, Layout(size)],
) raises:
    _, o = enqueue_create_matrix[
        layout = Layout(size, 1), dtype = DType.uint32
    ](ctx)

    fn _compare(t1: __type_of(t1), t2: __type_of(t2), out: __type_of(o)):
        i = block_idx.x
        eql = t1[i][0] == t2[i][0]
        out[i, 0] = eql.cast[DType.uint32]()

    ctx.enqueue_function[_compare](t1, t2, o, grid_dim=size, block_dim=1)

    fn print_acc(out: __type_of(o)):
        shared = stack_allocation[
            32, DType.uint32, address_space = AddressSpace.SHARED
        ]()
        shared.store(SIMD[DType.uint32, 32](0))

        i = thread_idx.x

        for shft in range(size // 1024):
            ii = shft * 1024 + i
            eql = 0 if ii >= size else out[ii, 0][0]
            sum = warp.sum(eql)
            shared[i] += sum

        barrier()
        if i == 0:
            print(shared.load[width=32]().reduce_add())

    ctx.enqueue_function[print_acc](o, grid_dim=1, block_dim=1024)
    ctx.synchronize()

    # host_buff = ctx.enqueue_create_host_buffer[dtype](1)
    # host_t = LayoutTensor[dtype, Layout(1)](host_buff)
    # ctx.synchronize()
    # host_t.copy_from(out_t)
    # return host_buff[0] / size
    # return 0
