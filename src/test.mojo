from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, barrier
from gpu import warp
from gpu.memory import AddressSpace
from memory import stack_allocation
from layout import LayoutTensor, Layout
from math import iota

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

    fn sum_zero_axis(
        tensor: LayoutTensor[dtype, layout, MutableAnyOrigin],
        out_tensor: __type_of(out_tensor),
    ):
        alias warps = cols // 32 + (cols if cols % 32 > 0 else 0)
        shared = stack_allocation[
            cols * warps,
            Scalar[dtype],
            address_space = AddressSpace.SHARED,
        ]()
        r, c = thread_idx.x, block_idx.x
        value = tensor.load[1](r, c)
        value = warp.sum(value)

        if block_idx.x == 0:
            print("thread:", thread_idx.x, "value:", value)

        if thread_idx.x == 0:
            out_tensor[c] = value

        barrier()

        if c == 0 and r == 0:
            print("[", end="")
            for i in range(cols):
                print(out_tensor[i], end=", ")
            print("]")

            print("It's ok?:", out_tensor[0] == 528)
            print("Warp working but short?:", out_tensor[0] == (528 - 32))
        #     print("===== OUTPUT =====")
        #     print(out_tensor)
        #     print("==================")

    ctx.enqueue_function[sum_zero_axis](
        device_tensor, out_tensor, grid_dim=cols, block_dim=rows
    )

    ctx.synchronize()

    # print(out_buff)
    # print(out_tensor)
