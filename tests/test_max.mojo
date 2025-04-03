from layout import Layout, LayoutTensor
from gpu.host import DeviceContext, DeviceBuffer
from gpu import thread_idx, block_idx, warp, barrier
from memory import stack_allocation
from gpu.memory import AddressSpace
from math import iota


fn main() raises:
    var ctx = DeviceContext()
    alias rows = 3
    alias cols = 4
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
    ctx: DeviceContext,
    ti: LayoutTensor[dtype, Layout.row_major(rows, cols)],
) raises -> Scalar[dtype]:
    host_buff = ctx.enqueue_create_host_buffer[dtype](1)

    fn all_max(
        rd: Int,
        cd: Int,
        t: __type_of(ti),
        final: __type_of(host_buff),
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
            host_buff,
            grid_dim=new_cols,
            block_dim=new_rows,
        )

        new_rows = 1

    ctx.synchronize()
    return host_buff[0]


# fn max[
#     dtype: DType, rows: Int, cols: Int, //
# ](
#     ctx: DeviceContext,
#     ti: LayoutTensor[dtype, Layout.row_major(rows, cols)],
# ) raises -> Scalar[dtype]:
#     host_buff = ctx.enqueue_create_host_buffer[dtype](1)

#     fn all_max(
#         t: __type_of(ti),
#         final: __type_of(host_buff),
#     ):
#         shared = stack_allocation[
#             32, dtype, address_space = AddressSpace.SHARED
#         ]()

#         r, c = thread_idx.x, block_idx.x

#         new_cols = cols
#         new_rows = rows
#         while new_cols != 1:
#             elems, new_rows = new_cols * new_rows, 1024
#             while elems % new_rows != 0:
#                 new_rows -= 1
#             new_cols = elems // new_rows

#             if r == 0 and c == 0:
#                 print(new_cols, new_rows)
#             # Here, only use threads and blocks needed.
#             # The rest could be released
#             if r > new_rows and c > new_cols:
#                 return

#             idx = r * new_cols + c
#             # Calculate row and col based on index.
#             rr, rc = idx // cols, idx % cols

#             tvalue = t[rr, rc][0]
#             value = warp.max(tvalue)
#             shared[r // 32] = value

#             barrier()

#             if thread_idx.x == 0:
#                 max = shared.load[width=32]().reduce_max()
#                 lr, lc = c // cols, c % cols
#                 t[lr, lc] = max

#             new_rows = 1

#         if r == 0 and c == 0:
#             final[0] = t[0, 0][0]

#     new_cols = cols
#     new_rows = rows
#     # # While we have more than 1 columns, let's do row reduction, and
#     # while new_cols != 1:
#     elems, new_rows = new_cols * new_rows, 1024
#     while elems % new_rows != 0:
#         new_rows -= 1
#     new_cols = elems // new_rows

#     ctx.enqueue_function[all_max](
#         ti,
#         host_buff,
#         grid_dim=new_rows,
#         block_dim=new_cols,
#     )

#     # new_rows = 1

#     ctx.synchronize()
#     return host_buff[0]
