from gpu.host import DeviceContext
from gpu import warp, block_idx, thread_idx, barrier
from math import iota
from memory import stack_allocation
from gpu.memory import AddressSpace
from layout import LayoutTensor, Layout, IntTuple

alias LY = Layout


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

    fn all_reduce(
        rd: Int,
        cd: Int,
        t: __type_of(ti),
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
        if r % 32 == 0:
            shared[r // 32] = value

        barrier()

        if thread_idx.x == 0:
            max = simd_op(shared.load[width=32]())
            lr, lc = c // cols, c % cols
            t[lr, lc] = max

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
            grid_dim=new_cols,
            block_dim=new_rows,
        )

        new_rows = 1

    fn to_other_tensor(t: __type_of(ti), o: __type_of(out_t)):
        o[0] = t[0, 0][0]

    ctx.enqueue_function[to_other_tensor](ti, out_t, block_dim=1, grid_dim=1)
    return out_t


fn main() raises:
    alias dtype = DType.float32

    alias rows = 3
    alias cols = 512
    alias size = rows * cols

    ctx = DeviceContext()

    host_buff = ctx.enqueue_create_host_buffer[dtype](size)
    iota(host_buff.unsafe_ptr(), size)

    buff = ctx.enqueue_create_buffer[dtype](size)
    buff.enqueue_copy_from(host_buff)
    t = LayoutTensor[dtype, Layout(IntTuple(rows, cols))](buff)

    out_t = matrix_reduce[warp.sum, SIMD.reduce_add](ctx, t)
    ctx.synchronize()

    fn print_value(
        t: LayoutTensor[dtype, Layout(IntTuple(3, 3)), MutableAnyOrigin]
    ):
        print(t)

    ctx.enqueue_function[print_value](out_t, grid_dim=1, block_dim=1)
    ctx.synchronize()
