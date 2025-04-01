from layout import Layout, LayoutTensor
from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, warp, barrier
from memory import stack_allocation
from gpu.memory import AddressSpace
from math import iota


fn main() raises:
    var ctx = DeviceContext()
    alias rows = 30
    alias cols = 30
    alias dtype = DType.float32
    h_buff = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    iota(h_buff.unsafe_ptr(), rows * cols)
    buff = ctx.enqueue_create_buffer[dtype](rows * cols)
    buff.enqueue_copy_from(h_buff)
    # print(h_buff)
    ti = LayoutTensor[dtype, Layout.row_major(rows, cols)](buff)

    value = max(ctx, ti)
    ctx.synchronize()
    print("got :", value)


fn max[
    dtype: DType, rows: Int, cols: Int, //
](
    ctx: DeviceContext, ti: LayoutTensor[dtype, Layout.row_major(rows, cols)]
) raises -> Scalar[dtype]:
    # out = ctx.enqueue_create_buffer[dtype](rows * cols)
    # out_t = LayoutTensor[dtype, Layout.row_major(cols * rows)](out)
    print("okk")
    # out.enqueue_copy_from(tib)

    fn all_max(
        rd: Int,
        cd: Int,
        t: __type_of(ti),
        mut final: Scalar[dtype],
    ):
        shared = stack_allocation[
            32, dtype, address_space = AddressSpace.SHARED
        ]()

        r, c = thread_idx.x, block_idx.x
        idx = r * cd + c
        # Calculate row and col based on index.
        rr, rc = idx // cols, idx % cols

        tvalue = t[rr, rc][0]
        value = warp.max(tvalue)
        shared[r // 32] = value

        barrier()

        if thread_idx.x == 0:
            max = shared.load[width=32]().reduce_max()
            lr, lc = c // cols, c % cols
            # print(shared.load[width=32]())
            print(lr, lc)
            t[lr, lc] = max

        # if r == 0 and c == 0:
        #     print(t)

        if r == 0 and c == 0 and cd == 1:
            final = t[0][0]

    new_cols = cols
    new_rows = rows
    final = Scalar[dtype]()
    while new_cols != 1:
        elems, new_rows = new_cols * new_rows, 1024
        while elems % new_rows != 0:
            new_rows -= 1
        new_cols = elems // new_rows
        print("elements:", elems, "rows:", new_rows, "cols:", new_cols)

        ctx.enqueue_function[all_max](
            new_rows,
            new_cols,
            ti,
            final,
            grid_dim=new_cols,
            block_dim=new_rows,
        )

        new_rows = 1
        ctx.synchronize()

    ctx.synchronize()
    return final
    # final = ctx.enqueue_create_host_buffer[dtype](1)
    # final.enqueue_copy_from(out.create_sub_buffer[dtype](0, 1))
    # print("Got:", final)
