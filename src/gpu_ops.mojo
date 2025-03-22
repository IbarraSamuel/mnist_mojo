from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, composition, IntTuple

alias N: Int = 2
alias M: Int = 4
alias dtype = DType.float32
alias orig_layout = Layout.row_major(N, M)
alias OrigTensor = LayoutTensor[dtype, orig_layout, MutableAnyOrigin]


fn transpose(
    tensor: LayoutTensor[dtype, orig_layout, MutableAnyOrigin]
) -> LayoutTensor[
    dtype,
    composition(
        orig_layout,
        Layout(
            IntTuple(tensor.shape[1](), tensor.shape[0]()),
            IntTuple(tensor.shape[0](), 1),
        ),
    ),
    MutableAnyOrigin,
]:
    tt = tensor.transpose()
    return tt
