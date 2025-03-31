from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, barrier
from gpu import warp
from gpu.memory import AddressSpace
from memory import stack_allocation
from layout import LayoutTensor, Layout
from math import iota, ceil

from bit import next_power_of_two


fn main() raises:
    alias dtype = DType.float32
    alias rows = 33
    alias cols = 5
    alias size = rows * cols
    alias layout = Layout.col_major(rows, cols)
    ctx = DeviceContext()
    host_buffer = ctx.enqueue_create_host_buffer[dtype](size)
    iota(host_buffer.unsafe_ptr(), size)
    host_tensor = LayoutTensor[dtype, layout](host_buffer)
    print("====== INPUT =====")
    print(host_buffer)
    print(host_tensor)
    print("==================")

    device_buffer = ctx.enqueue_create_buffer[dtype](size)
    device_buffer.enqueue_copy_from(host_buffer)
    device_tensor = LayoutTensor[dtype, layout](device_buffer)

    ctx.synchronize()

    out_buff = ctx.enqueue_create_buffer[dtype](cols)
    out_buff = out_buff.enqueue_fill(0)
    out_tensor = LayoutTensor[dtype, Layout(cols)](out_buff)

    alias warps = rows // 32 + (1 if rows % 32 > 0 else 0)

    fn sum_zero_axis(
        tensor: LayoutTensor[dtype, layout, MutableAnyOrigin],
        out_tensor: __type_of(out_tensor),
    ):
        # Store each warp in consecutive spaces
        shared = stack_allocation[
            warps,
            Scalar[dtype],
            address_space = AddressSpace.SHARED,
        ]()

        r, c = thread_idx.x, block_idx.x
        tvalue = tensor.load[1](r, c)

        value = warp.sum(tvalue)
        warp_idx = r // 32
        shared[warp_idx] = value

        barrier()

        if thread_idx.x == 0:
            out_tensor[c] = shared.load[
                width = next_power_of_two(warps)
            ]().reduce_add()

    ctx.enqueue_function[sum_zero_axis](
        device_tensor, out_tensor, grid_dim=cols, block_dim=rows
    )

    ctx.synchronize()

    # print(out_buff)
    # print(out_tensor)
