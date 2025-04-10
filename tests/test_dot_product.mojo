from layout import Layout, LayoutTensor
from gpu.host import DeviceContext, DeviceBuffer
from gpu import thread_idx, block_idx, warp, barrier
from math import iota
from memory import stack_allocation
from gpu.memory import AddressSpace
from bit import next_power_of_two

from gpu_ops import import dot


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
