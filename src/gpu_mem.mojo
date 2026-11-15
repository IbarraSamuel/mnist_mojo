from std.math import sqrt
from std.memory import unsafe_memcpy
from std.random import randn
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import TileTensor, row_major


struct DeviceMatrix[dtype: DType, rows: Int, cols: Int](Movable):
    """Owning 2D matrix backed by a GPU DeviceBuffer."""

    comptime layout = row_major[Self.rows, Self.cols]()
    comptime num_elements = Self.rows * Self.cols

    var buffer: DeviceBuffer[Self.dtype]

    def __init__(
        out self, ctx: DeviceContext, fill: Scalar[Self.dtype] = 0
    ) raises:
        self.buffer = ctx.enqueue_create_buffer[Self.dtype](Self.num_elements)
        self.buffer.enqueue_fill(fill)

    def init_he(mut self, ctx: DeviceContext, fan_in: Int) raises:
        """He normal initialization on host, mapped into the device buffer."""
        comptime assert (
            Self.dtype.is_floating_point()
        ), "init_he requires floating-point dtype"
        var std_dev = sqrt(2.0 / Float64(fan_in))
        with self.buffer.map_to_host() as host:
            randn(host.as_span(), mean=0.0, standard_deviation=std_dev)

    def copy_from_host(mut self, span: Span[Scalar[Self.dtype], _]) raises:
        """Copies `span` into the device buffer via host mapping."""
        if len(span) < Self.num_elements:
            raise Error("DeviceMatrix.copy_from_host: source span too small")
        with self.buffer.map_to_host() as host:
            unsafe_memcpy(
                dest=host.unsafe_ptr(),
                src=span.unsafe_ptr(),
                count=Self.num_elements,
            )

    def copy_to_host(self, mut dst: List[Scalar[Self.dtype]]) raises:
        """Copies the device buffer into `dst`."""
        if len(dst) < Self.num_elements:
            raise Error("DeviceMatrix.copy_to_host: destination list too small")
        with self.buffer.map_to_host() as host:
            unsafe_memcpy(
                dest=dst.unsafe_ptr(),
                src=host.unsafe_ptr(),
                count=Self.num_elements,
            )

    @inline(.always)
    def fill(mut self, value: Scalar[Self.dtype]) raises:
        self.buffer.enqueue_fill(value)
