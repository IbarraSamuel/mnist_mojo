from layout import Layout, LayoutTensor
from gpu.host import DeviceContext, DeviceBuffer
from gpu import thread_idx, block_idx, warp, barrier
from math import iota
from memory import stack_allocation
from gpu.memory import AddressSpace
from bit import next_power_of_two


fn dot[
    x: Int, y: Int, z: Int, dtype: DType
](
    ctx: DeviceContext,
    t1: LayoutTensor[dtype, Layout.row_major(x, y)],
    t2: LayoutTensor[dtype, Layout.row_major(y, z)],
    mut tob: DeviceBuffer[dtype],
    mut to: LayoutTensor[dtype, Layout.row_major(x, z)],
) raises:
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

    # return tob, to


fn main() raises:
    ctx = DeviceContext()
    alias rows_1 = 2
    alias cols_1 = 3
    alias rows_2 = 3
    alias cols_2 = 4
    alias dtype = DType.float32

    host_buff = ctx.enqueue_create_host_buffer[dtype](
        max(rows_1, rows_2) * max(cols_1, cols_2)
    )
    iota(host_buff.unsafe_ptr(), len(host_buff))

    buff = ctx.enqueue_create_buffer[dtype](rows_1 * cols_1)
    buff.enqueue_copy_from(host_buff.create_sub_buffer[dtype](0, len(buff)))
    t1 = LayoutTensor[dtype, Layout.row_major(rows_1, cols_1)](buff)

    buff2 = ctx.enqueue_create_buffer[dtype](rows_2 * cols_2)
    buff2.enqueue_copy_from(host_buff.create_sub_buffer[dtype](0, len(buff2)))
    t2 = LayoutTensor[dtype, Layout.row_major(rows_2, cols_2)](buff2)

    buffo = ctx.enqueue_create_buffer[dtype](rows_1 * cols_2)
    to = LayoutTensor[dtype, Layout.row_major(rows_1, cols_2)](buffo)

    dot(ctx, t1, t2, buffo, to)
