from layout import Layout, LayoutTensor, IntTuple
from gpu.host import DeviceContext, DeviceBuffer
from gpu import thread_idx, block_idx, warp, barrier
from math import iota
from memory import stack_allocation
from gpu.memory import AddressSpace
from bit import next_power_of_two

from gpu_ops import dot


fn dot_large[
    dtype: DType,
    ly1: Layout,
    ly2: Layout,
](
    ctx: DeviceContext,
    t1: LayoutTensor[dtype, ly1],
    t2: LayoutTensor[dtype, ly2],
) raises -> LayoutTensor[
    dtype, Layout(IntTuple(t1.shape[0](), t2.shape[1]())), MutableAnyOrigin
]:
    alias x_dim = t1.shape[0]()
    alias z_dim = t1.shape[1]()
    alias y_dim = t2.shape[1]()
    constrained[z_dim == t2.shape[0](), "Dims should match"]()

    buff = ctx.enqueue_create_buffer[dtype](x_dim * y_dim * z_dim)

    out = LayoutTensor[
        dtype, Layout(IntTuple(x_dim, z_dim, y_dim)), MutableAnyOrigin
    ](buff)

    fn dot_large_gpu(t1: __type_of(t1), t2: __type_of(t2), o: __type_of(out)):
        x, z, y = thread_idx.x, block_idx.x, thread_idx.y
        o[x, z, y] = t1[x, z] * t2[z, y]

    ctx.enqueue_function[dot_large_gpu](
        t1, t2, out, grid_dim=z_dim, block_dim=(x_dim, y_dim)
    )

    alias warps_per_v = z_dim // 32 + (
        1 if z_dim % 32 > 0 else 0
    )  # Warps number
    alias wpv_2pow = next_power_of_two(warps_per_v)
    alias repeat_threads = warps_per_v // 32 + (
        1 if warps_per_v % 32 > 0 else 0
    )  # loops in the same thread
    alias block_dim = min(1024, z_dim)
    """If the z dim is very small, don't need to do it bigger than we really need."""

    obuff = ctx.enqueue_create_buffer[dtype](x_dim * y_dim)
    obuff = obuff.enqueue_fill(0)
    out2 = LayoutTensor[
        dtype, Layout(IntTuple(x_dim, y_dim)), MutableAnyOrigin
    ](obuff)

    fn reduce_with_warps(o: __type_of(out), o2: __type_of(out2)):
        shared = stack_allocation[
            warps_per_v, dtype, address_space = AddressSpace.SHARED
        ]()
        x, z, y = block_idx.x, thread_idx.x, block_idx.y

        for blk in range(repeat_threads):
            # Move the thread 1024 down to calc next portion
            zb = 1024 * blk + z
            # get the value out ot the tensor
            sval = o[x, zb, y][0]
            # warp into a single value
            tot = warp.sum(sval)

            # save into the shared location
            warp_idx = zb // 32
            shared[warp_idx] = tot

        barrier()
        if z == 0:
            final = (
                shared.load[width=wpv_2pow]()
                .shift_left[wpv_2pow - warps_per_v]()
                .reduce_add()
            )
            o2[x, y] = final

    ctx.enqueue_function[reduce_with_warps](
        out, out2, grid_dim=(x_dim, y_dim), block_dim=block_dim
    )

    return out2


fn main() raises:
    ctx = DeviceContext()
    alias rows_1 = 2
    alias cols_1 = 9
    alias rows_2 = 9
    alias cols_2 = 3
    alias dtype = DType.float32

    host_buff = ctx.enqueue_create_host_buffer[dtype](
        max(rows_1, rows_2) * max(cols_1, cols_2)
    )
    iota(host_buff.unsafe_ptr(), len(host_buff))

    buff = ctx.enqueue_create_buffer[dtype](rows_1 * cols_1)
    buff.enqueue_copy_from(host_buff.create_sub_buffer[dtype](0, len(buff)))
    t1 = LayoutTensor[dtype, Layout(IntTuple(rows_1, cols_1))](buff)

    buff2 = ctx.enqueue_create_buffer[dtype](rows_2 * cols_2)
    buff2.enqueue_copy_from(host_buff.create_sub_buffer[dtype](0, len(buff2)))
    t2 = LayoutTensor[dtype, Layout(IntTuple(rows_2, cols_2))](buff2)

    res2 = dot_large(ctx, t1, t2)
    res1 = dot(ctx, t1, t2)

    fn print_tensor(t: __type_of(res1)):
        print(t, "\n")

    # fn print_tensor_2(t: __type_of(res2)):
    #     print(t, "\n")

    ctx.enqueue_function[print_tensor](res1, grid_dim=1, block_dim=1)
    ctx.enqueue_function[print_tensor](res2, grid_dim=1, block_dim=1)
    ctx.synchronize()
