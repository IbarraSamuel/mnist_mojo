from gpu.host import DeviceContext
from gpu import warp, block_idx, thread_idx, barrier
from gpu.id import block_dim
from math import iota
from bit import next_power_of_two
from memory import stack_allocation
from gpu.memory import AddressSpace
from layout import LayoutTensor, Layout, IntTuple

alias LY = Layout


# fn matrix_reduce[
#     dtype: DType,
#     ly: LY, //,
#     warp_op: fn[dt: DType, w: Int, //] (SIMD[dt, w]) -> SIMD[dt, w],
#     simd_op: fn[d: DType, s: Int] (SIMD[d, s]) -> SIMD[d, 1],
# ](
#     ctx: DeviceContext,
#     ti: LayoutTensor[dtype, ly],
# ) raises -> LayoutTensor[
#     dtype, Layout(1), MutableAnyOrigin
# ]:
#     alias rows = ti.shape[0]()
#     alias cols = ti.shape[1]()
#     alias size = rows * cols

#     alias block_max = 1024
#     alias loop_size = size // block_max + (1 if size % block_max > 0 else 0)

#     alias load_size = next_power_of_two(loop_size)
#     alias threads = size // load_size + (1 if size % load_size > 0 else 0)

#     print(load_size, threads)
#     buff = ctx.enqueue_create_buffer[dtype](1)
#     out_t = LayoutTensor[dtype, Layout(1), MutableAnyOrigin](buff)

#     fn calc_red(t: __type_of(ti), o: __type_of(out_t)):
#         shared = stack_allocation[
#             32, dtype, address_space = AddressSpace.SHARED
#         ]()

#         i = thread_idx.x
#         start = i * load_size
#         r, c = start % rows, start // rows
#         simd = t.load[load_size](r, c)

#         if i == block_dim.x - 1:
#             simd = simd.shift_left[threads * load_size - size]()

#         accum = simd_op(simd)
#         wacc = warp_op(accum)

#         if i % 32 == 0:
#             shared[i // 32] = wacc

#         barrier()
#         if i == 0:
#             o[0] = shared.load[width=32]().reduce_add()

#     ctx.enqueue_function[calc_red](ti, out_t, grid_dim=1, block_dim=threads)
#     return out_t


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

    buff = ctx.enqueue_create_buffer[dtype](1)
    out_t = LayoutTensor[dtype, Layout(1), MutableAnyOrigin](buff)
    # _, out_t = enqueue_create_matrix[size=1, dtype=dtype](ctx)

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

        tvalue = t.load[1](rr, rc)
        value = warp_op(tvalue)
        shared[r // 32] = value

        barrier()

        if thread_idx.x == 0:
            max = simd_op(shared.load[width=32]())
            lr, lc = c // cols, c % cols
            t[lr, lc] = max

        if thread_idx.x == 0 and block_idx.x == 0 and cd == 1:
            final[0, 0] = t.load[1](0, 0)

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


fn main() raises:
    alias dtype = DType.float32

    alias rows = 512
    alias cols = 821
    alias size = rows * cols

    ctx = DeviceContext()

    host_buff = ctx.enqueue_create_host_buffer[dtype](size)
    iota(host_buff.unsafe_ptr(), size)
    tot = Scalar[dtype](0)

    print("Manual count:")
    for i in range(size):
        tot = max(host_buff[i], tot)
    print(tot)

    print("Other count:")
    buff = ctx.enqueue_create_buffer[dtype](size)
    buff.enqueue_copy_from(host_buff)
    t = LayoutTensor[dtype, Layout(IntTuple(rows, cols))](buff)

    out_t = matrix_reduce[warp.max, SIMD.reduce_max](ctx, t)
    ctx.synchronize()

    lbuff = ctx.enqueue_create_host_buffer[dtype](1)
    l = LayoutTensor[dtype, Layout(1)](lbuff)

    fn to_local(
        g: LayoutTensor[dtype, Layout(1), MutableAnyOrigin],
        l: LayoutTensor[dtype, Layout(1), MutableAnyOrigin],
    ):
        l[0] = g[0]

    ctx.enqueue_function[to_local](out_t, l, grid_dim=1, block_dim=1)

    ctx.synchronize()
    print(l)
