from std.algorithm import vectorize
from std.collections import Array
from std.math import ceildiv, exp, log, max, sqrt
from std.memory import unsafe_memcpy
from std.random import randn
from std.sys import simd_width_of
from max.algorithm import parallelize


comptime SIMD_W = simd_width_of[DType.float32]()
comptime SIMD_2W = 2 * SIMD_W
comptime SIMD_4W = 4 * SIMD_W
comptime PARALLEL_FLOP_THRESHOLD = 500_000
comptime PARALLEL_ELEM_THRESHOLD = 32_768

comptime EPILOGUE_NONE = 0
comptime EPILOGUE_BIAS = 1
comptime EPILOGUE_BIAS_RELU = 2
comptime EPILOGUE_BIAS_ROW = 3
comptime EPILOGUE_BIAS_ROW_RELU = 4
comptime EPILOGUE_RELU_MASK = 5


@fieldwise_init
struct MatrixView[mut: Bool, //, origin: Origin[mut=mut]](
    TrivialRegisterPassable
):
    """Non-owning register-passable 2D Float32 matrix view."""

    var ptr: Pointer[Float32, Self.origin]
    var rows: Int
    var cols: Int

    @inline(.always)
    def numel(self) -> Int:
        return self.rows * self.cols

    @inline(.always)
    def row_ptr(
        self, r: Int
    ) -> Pointer[Float32, mut=Self.mut, origin=Self.origin]:
        return self.ptr.unsafe_offset(r * self.cols)

    @inline(.always)
    def load[w: Int = 1](self, r: Int, c: Int) -> SIMD[DType.float32, w]:
        return self.ptr.unsafe_load[width=w](r * self.cols + c)

    @inline(.always)
    def store[
        w: Int = 1
    ](
        self: MatrixView[mut=True, origin=Self.origin],
        r: Int,
        c: Int,
        val: SIMD[DType.float32, w],
    ):
        self.ptr.unsafe_store[width=w](r * self.cols + c, val)


@fieldwise_init
struct StaticMatrixView[
    mut: Bool, //, rows: Int, cols: Int, origin: Origin[mut=mut]
](TrivialRegisterPassable):
    """Zero-overhead register-passable 2D view with compile-time dimensions."""

    comptime numel = Self.rows * Self.cols

    var ptr: Pointer[Float32, Self.origin]

    @inline(.always)
    def row_ptr(
        self, r: Int
    ) -> Pointer[Float32, mut=Self.mut, origin=Self.origin]:
        return self.ptr.unsafe_offset(r * Self.cols)


struct CPUMatrix(Copyable):
    """Owning row-major Float32 matrix on host memory."""

    var rows: Int
    var cols: Int
    var data: List[Float32]

    def __init__(out self, rows: Int, cols: Int, fill: Float32 = 0.0):
        self.rows = rows
        self.cols = cols
        self.data = List[Float32](length=rows * cols, fill=fill)

    @inline(.always)
    def zero(mut self):
        var ptr = self.data.unsafe_ptr()
        var n = len(self.data)

        @inline(.always)
        def _zero[w: Int](i: Int) {imm ptr}:
            ptr.unsafe_store[width=w](i, SIMD[DType.float32, w](0.0))

        vectorize[SIMD_W](n, _zero)

    def init_he(mut self, fan_in: Int):
        """He normal initialization: N(0, sqrt(2 / fan_in))."""
        from std.math import sqrt

        var std_dev = sqrt(2.0 / Float64(fan_in))
        randn(Span(self.data), mean=0.0, standard_deviation=std_dev)

    @inline(.always)
    def ptr(self) -> ImmPointer[Float32, origin_of(self.data)]:
        return self.data.unsafe_ptr()

    @inline(.always)
    def mut_ptr(mut self) -> MutPointer[Float32, origin_of(self.data)]:
        return self.data.unsafe_ptr()

    @inline(.always)
    def view(self) -> MatrixView[origin_of(self.data)]:
        return MatrixView(self.data.unsafe_ptr(), self.rows, self.cols)

    @inline(.always)
    def mut_view(mut self) -> MatrixView[origin_of(self.data)]:
        return MatrixView(self.data.unsafe_ptr(), self.rows, self.cols)

    @inline(.always)
    def static_view[
        r: Int, c: Int
    ](self) -> StaticMatrixView[r, c, origin_of(self.data)]:
        return StaticMatrixView[r, c](self.data.unsafe_ptr())

    @inline(.always)
    def static_mut_view[
        r: Int, c: Int
    ](mut self) -> StaticMatrixView[r, c, origin_of(self.data)]:
        return StaticMatrixView[r, c](self.data.unsafe_ptr())


@inline(.always)
def _store_nn_tile_row[
    epilogue: Int, w: Int
](
    c_row: MutPointer[Float32, _],
    aux_row: ImmPointer[Float32, _],
    j: Int,
    acc: SIMD[DType.float32, w],
    bias_val: Float32,
):
    var res = acc
    comptime if epilogue == EPILOGUE_BIAS:
        res = res + bias_val
    elif epilogue == EPILOGUE_BIAS_RELU:
        res = max(res + bias_val, SIMD[DType.float32, w](0.0))
    elif epilogue == EPILOGUE_BIAS_ROW:
        res = res + aux_row.unsafe_load[width=w](j)
    elif epilogue == EPILOGUE_BIAS_ROW_RELU:
        res = max(
            res + aux_row.unsafe_load[width=w](j),
            SIMD[DType.float32, w](0.0),
        )
    elif epilogue == EPILOGUE_RELU_MASK:
        var m = aux_row.unsafe_load[width=w](j)
        res = m.gt(0.0).select(res, SIMD[DType.float32, w](0.0))
    c_row.unsafe_store[width=w](j, res)


@inline(.always)
def _matmul_nn_strip4x4[
    K_s: Int, epilogue: Int
](
    ap: ImmPointer[Float32, _],
    bp: ImmPointer[Float32, _],
    cp: MutPointer[Float32, _],
    bias_p: ImmPointer[Float32, _],
    i: Int,
    j: Int,
    N: Int,
    K: Int,
):
    """Computes a 4 x SIMD_4W tile [i..i+3, j..j+SIMD_4W-1] of C = A @ B (+ epilogue).
    """
    var a0p = ap.unsafe_offset(i * K)
    var a1p = ap.unsafe_offset((i + 1) * K)
    var a2p = ap.unsafe_offset((i + 2) * K)
    var a3p = ap.unsafe_offset((i + 3) * K)
    var c0p = cp.unsafe_offset(i * N)
    var c1p = cp.unsafe_offset((i + 1) * N)
    var c2p = cp.unsafe_offset((i + 2) * N)
    var c3p = cp.unsafe_offset((i + 3) * N)

    var b0_val = Float32(0.0)
    var b1_val = Float32(0.0)
    var b2_val = Float32(0.0)
    var b3_val = Float32(0.0)
    comptime if epilogue == EPILOGUE_BIAS or epilogue == EPILOGUE_BIAS_RELU:
        b0_val = bias_p[unsafe_offset=i]
        b1_val = bias_p[unsafe_offset=i + 1]
        b2_val = bias_p[unsafe_offset=i + 2]
        b3_val = bias_p[unsafe_offset=i + 3]

    var c00 = SIMD[DType.float32, SIMD_W](0.0)
    var c01 = SIMD[DType.float32, SIMD_W](0.0)
    var c02 = SIMD[DType.float32, SIMD_W](0.0)
    var c03 = SIMD[DType.float32, SIMD_W](0.0)
    var c10 = SIMD[DType.float32, SIMD_W](0.0)
    var c11 = SIMD[DType.float32, SIMD_W](0.0)
    var c12 = SIMD[DType.float32, SIMD_W](0.0)
    var c13 = SIMD[DType.float32, SIMD_W](0.0)
    var c20 = SIMD[DType.float32, SIMD_W](0.0)
    var c21 = SIMD[DType.float32, SIMD_W](0.0)
    var c22 = SIMD[DType.float32, SIMD_W](0.0)
    var c23 = SIMD[DType.float32, SIMD_W](0.0)
    var c30 = SIMD[DType.float32, SIMD_W](0.0)
    var c31 = SIMD[DType.float32, SIMD_W](0.0)
    var c32 = SIMD[DType.float32, SIMD_W](0.0)
    var c33 = SIMD[DType.float32, SIMD_W](0.0)

    comptime if K_s > 0 and K_s <= 16:
        comptime for k in range(K_s):
            var bk = bp.unsafe_offset(k * N + j)
            var bv0 = bk.unsafe_load[width=SIMD_W](0)
            var bv1 = bk.unsafe_load[width=SIMD_W](SIMD_W)
            var bv2 = bk.unsafe_load[width=SIMD_W](2 * SIMD_W)
            var bv3 = bk.unsafe_load[width=SIMD_W](3 * SIMD_W)
            var a0 = SIMD[DType.float32, SIMD_W](a0p[unsafe_offset=k])
            var a1 = SIMD[DType.float32, SIMD_W](a1p[unsafe_offset=k])
            var a2 = SIMD[DType.float32, SIMD_W](a2p[unsafe_offset=k])
            var a3 = SIMD[DType.float32, SIMD_W](a3p[unsafe_offset=k])
            c00 = a0.fma(bv0, c00)
            c01 = a0.fma(bv1, c01)
            c02 = a0.fma(bv2, c02)
            c03 = a0.fma(bv3, c03)
            c10 = a1.fma(bv0, c10)
            c11 = a1.fma(bv1, c11)
            c12 = a1.fma(bv2, c12)
            c13 = a1.fma(bv3, c13)
            c20 = a2.fma(bv0, c20)
            c21 = a2.fma(bv1, c21)
            c22 = a2.fma(bv2, c22)
            c23 = a2.fma(bv3, c23)
            c30 = a3.fma(bv0, c30)
            c31 = a3.fma(bv1, c31)
            c32 = a3.fma(bv2, c32)
            c33 = a3.fma(bv3, c33)
    else:
        for k in range(K):
            var bk = bp.unsafe_offset(k * N + j)
            var bv0 = bk.unsafe_load[width=SIMD_W](0)
            var bv1 = bk.unsafe_load[width=SIMD_W](SIMD_W)
            var bv2 = bk.unsafe_load[width=SIMD_W](2 * SIMD_W)
            var bv3 = bk.unsafe_load[width=SIMD_W](3 * SIMD_W)
            var a0 = SIMD[DType.float32, SIMD_W](a0p[unsafe_offset=k])
            var a1 = SIMD[DType.float32, SIMD_W](a1p[unsafe_offset=k])
            var a2 = SIMD[DType.float32, SIMD_W](a2p[unsafe_offset=k])
            var a3 = SIMD[DType.float32, SIMD_W](a3p[unsafe_offset=k])
            c00 = a0.fma(bv0, c00)
            c01 = a0.fma(bv1, c01)
            c02 = a0.fma(bv2, c02)
            c03 = a0.fma(bv3, c03)
            c10 = a1.fma(bv0, c10)
            c11 = a1.fma(bv1, c11)
            c12 = a1.fma(bv2, c12)
            c13 = a1.fma(bv3, c13)
            c20 = a2.fma(bv0, c20)
            c21 = a2.fma(bv1, c21)
            c22 = a2.fma(bv2, c22)
            c23 = a2.fma(bv3, c23)
            c30 = a3.fma(bv0, c30)
            c31 = a3.fma(bv1, c31)
            c32 = a3.fma(bv2, c32)
            c33 = a3.fma(bv3, c33)

    var aux0 = (
        bias_p.unsafe_offset(i * N) if epilogue
        == EPILOGUE_RELU_MASK else bias_p
    )
    var aux1 = (
        bias_p.unsafe_offset((i + 1) * N) if epilogue
        == EPILOGUE_RELU_MASK else bias_p
    )
    var aux2 = (
        bias_p.unsafe_offset((i + 2) * N) if epilogue
        == EPILOGUE_RELU_MASK else bias_p
    )
    var aux3 = (
        bias_p.unsafe_offset((i + 3) * N) if epilogue
        == EPILOGUE_RELU_MASK else bias_p
    )

    _store_nn_tile_row[epilogue, SIMD_W](c0p, aux0, j, c00, b0_val)
    _store_nn_tile_row[epilogue, SIMD_W](c0p, aux0, j + SIMD_W, c01, b0_val)
    _store_nn_tile_row[epilogue, SIMD_W](c0p, aux0, j + 2 * SIMD_W, c02, b0_val)
    _store_nn_tile_row[epilogue, SIMD_W](c0p, aux0, j + 3 * SIMD_W, c03, b0_val)
    _store_nn_tile_row[epilogue, SIMD_W](c1p, aux1, j, c10, b1_val)
    _store_nn_tile_row[epilogue, SIMD_W](c1p, aux1, j + SIMD_W, c11, b1_val)
    _store_nn_tile_row[epilogue, SIMD_W](c1p, aux1, j + 2 * SIMD_W, c12, b1_val)
    _store_nn_tile_row[epilogue, SIMD_W](c1p, aux1, j + 3 * SIMD_W, c13, b1_val)
    _store_nn_tile_row[epilogue, SIMD_W](c2p, aux2, j, c20, b2_val)
    _store_nn_tile_row[epilogue, SIMD_W](c2p, aux2, j + SIMD_W, c21, b2_val)
    _store_nn_tile_row[epilogue, SIMD_W](c2p, aux2, j + 2 * SIMD_W, c22, b2_val)
    _store_nn_tile_row[epilogue, SIMD_W](c2p, aux2, j + 3 * SIMD_W, c23, b2_val)
    _store_nn_tile_row[epilogue, SIMD_W](c3p, aux3, j, c30, b3_val)
    _store_nn_tile_row[epilogue, SIMD_W](c3p, aux3, j + SIMD_W, c31, b3_val)
    _store_nn_tile_row[epilogue, SIMD_W](c3p, aux3, j + 2 * SIMD_W, c32, b3_val)
    _store_nn_tile_row[epilogue, SIMD_W](c3p, aux3, j + 3 * SIMD_W, c33, b3_val)


@inline(.always)
def _matmul_nn_row4[
    M_s: Int, N_s: Int, K_s: Int, epilogue: Int
](
    ap: ImmPointer[Float32, _],
    bp: ImmPointer[Float32, _],
    cp: MutPointer[Float32, _],
    bias_p: ImmPointer[Float32, _],
    i: Int,
    N: Int,
    K: Int,
):
    """Computes 4 output rows [i..i+3] of C = A @ B (+ optional epilogue) in registers.
    """
    var a0p = ap.unsafe_offset(i * K)
    var a1p = ap.unsafe_offset((i + 1) * K)
    var a2p = ap.unsafe_offset((i + 2) * K)
    var a3p = ap.unsafe_offset((i + 3) * K)
    var c0p = cp.unsafe_offset(i * N)
    var c1p = cp.unsafe_offset((i + 1) * N)
    var c2p = cp.unsafe_offset((i + 2) * N)
    var c3p = cp.unsafe_offset((i + 3) * N)
    var aux0 = (
        bias_p.unsafe_offset(i * N) if epilogue
        == EPILOGUE_RELU_MASK else bias_p
    )
    var aux1 = (
        bias_p.unsafe_offset((i + 1) * N) if epilogue
        == EPILOGUE_RELU_MASK else bias_p
    )
    var aux2 = (
        bias_p.unsafe_offset((i + 2) * N) if epilogue
        == EPILOGUE_RELU_MASK else bias_p
    )
    var aux3 = (
        bias_p.unsafe_offset((i + 3) * N) if epilogue
        == EPILOGUE_RELU_MASK else bias_p
    )

    var b0_val = Float32(0.0)
    var b1_val = Float32(0.0)
    var b2_val = Float32(0.0)
    var b3_val = Float32(0.0)
    comptime if epilogue == EPILOGUE_BIAS or epilogue == EPILOGUE_BIAS_RELU:
        b0_val = bias_p[unsafe_offset=i]
        b1_val = bias_p[unsafe_offset=i + 1]
        b2_val = bias_p[unsafe_offset=i + 2]
        b3_val = bias_p[unsafe_offset=i + 3]

    var n_vec4_end = (N // SIMD_4W) * SIMD_4W
    var n_vec2_end = (N // SIMD_2W) * SIMD_2W
    var n_vec_end = (N // SIMD_W) * SIMD_W

    var j = 0
    comptime if SIMD_W <= 4:
        while j < n_vec4_end:
            _matmul_nn_strip4x4[K_s, epilogue](ap, bp, cp, bias_p, i, j, N, K)
            j += SIMD_4W

    while j < n_vec2_end:
        var c00 = SIMD[DType.float32, SIMD_W](0.0)
        var c01 = SIMD[DType.float32, SIMD_W](0.0)
        var c10 = SIMD[DType.float32, SIMD_W](0.0)
        var c11 = SIMD[DType.float32, SIMD_W](0.0)
        var c20 = SIMD[DType.float32, SIMD_W](0.0)
        var c21 = SIMD[DType.float32, SIMD_W](0.0)
        var c30 = SIMD[DType.float32, SIMD_W](0.0)
        var c31 = SIMD[DType.float32, SIMD_W](0.0)

        for k in range(K):
            var bv0 = bp.unsafe_load[width=SIMD_W](k * N + j)
            var bv1 = bp.unsafe_load[width=SIMD_W](k * N + j + SIMD_W)
            var a0 = SIMD[DType.float32, SIMD_W](a0p[unsafe_offset=k])
            var a1 = SIMD[DType.float32, SIMD_W](a1p[unsafe_offset=k])
            var a2 = SIMD[DType.float32, SIMD_W](a2p[unsafe_offset=k])
            var a3 = SIMD[DType.float32, SIMD_W](a3p[unsafe_offset=k])
            c00 = a0.fma(bv0, c00)
            c01 = a0.fma(bv1, c01)
            c10 = a1.fma(bv0, c10)
            c11 = a1.fma(bv1, c11)
            c20 = a2.fma(bv0, c20)
            c21 = a2.fma(bv1, c21)
            c30 = a3.fma(bv0, c30)
            c31 = a3.fma(bv1, c31)

        _store_nn_tile_row[epilogue, SIMD_W](c0p, aux0, j, c00, b0_val)
        _store_nn_tile_row[epilogue, SIMD_W](c0p, aux0, j + SIMD_W, c01, b0_val)
        _store_nn_tile_row[epilogue, SIMD_W](c1p, aux1, j, c10, b1_val)
        _store_nn_tile_row[epilogue, SIMD_W](c1p, aux1, j + SIMD_W, c11, b1_val)
        _store_nn_tile_row[epilogue, SIMD_W](c2p, aux2, j, c20, b2_val)
        _store_nn_tile_row[epilogue, SIMD_W](c2p, aux2, j + SIMD_W, c21, b2_val)
        _store_nn_tile_row[epilogue, SIMD_W](c3p, aux3, j, c30, b3_val)
        _store_nn_tile_row[epilogue, SIMD_W](c3p, aux3, j + SIMD_W, c31, b3_val)
        j += SIMD_2W

    comptime if N_s < 0 or (N_s % SIMD_2W) >= SIMD_W:
        while j < n_vec_end:
            var c0 = SIMD[DType.float32, SIMD_W](0.0)
            var c1 = SIMD[DType.float32, SIMD_W](0.0)
            var c2 = SIMD[DType.float32, SIMD_W](0.0)
            var c3 = SIMD[DType.float32, SIMD_W](0.0)

            for k in range(K):
                var bv = bp.unsafe_load[width=SIMD_W](k * N + j)
                c0 = SIMD[DType.float32, SIMD_W](a0p[unsafe_offset=k]).fma(
                    bv, c0
                )
                c1 = SIMD[DType.float32, SIMD_W](a1p[unsafe_offset=k]).fma(
                    bv, c1
                )
                c2 = SIMD[DType.float32, SIMD_W](a2p[unsafe_offset=k]).fma(
                    bv, c2
                )
                c3 = SIMD[DType.float32, SIMD_W](a3p[unsafe_offset=k]).fma(
                    bv, c3
                )

            _store_nn_tile_row[epilogue, SIMD_W](c0p, aux0, j, c0, b0_val)
            _store_nn_tile_row[epilogue, SIMD_W](c1p, aux1, j, c1, b1_val)
            _store_nn_tile_row[epilogue, SIMD_W](c2p, aux2, j, c2, b2_val)
            _store_nn_tile_row[epilogue, SIMD_W](c3p, aux3, j, c3, b3_val)
            j += SIMD_W

    comptime if N_s < 0 or (N_s % SIMD_W) != 0:
        while j < N:
            var s0 = Float32(0.0)
            var s1 = Float32(0.0)
            var s2 = Float32(0.0)
            var s3 = Float32(0.0)
            for k in range(K):
                var bv = bp[unsafe_offset=k * N + j]
                s0 = a0p[unsafe_offset=k] * bv + s0
                s1 = a1p[unsafe_offset=k] * bv + s1
                s2 = a2p[unsafe_offset=k] * bv + s2
                s3 = a3p[unsafe_offset=k] * bv + s3
            _store_nn_tile_row[epilogue, 1](c0p, aux0, j, s0, b0_val)
            _store_nn_tile_row[epilogue, 1](c1p, aux1, j, s1, b1_val)
            _store_nn_tile_row[epilogue, 1](c2p, aux2, j, s2, b2_val)
            _store_nn_tile_row[epilogue, 1](c3p, aux3, j, s3, b3_val)
            j += 1


@inline(.always)
def _matmul_nn_row1[
    N_s: Int, epilogue: Int
](
    ap: ImmPointer[Float32, _],
    bp: ImmPointer[Float32, _],
    cp: MutPointer[Float32, _],
    bias_p: ImmPointer[Float32, _],
    i: Int,
    N: Int,
    K: Int,
):
    """Computes a single output row i of C = A @ B (+ optional epilogue)."""
    var a_row = ap.unsafe_offset(i * K)
    var c_row = cp.unsafe_offset(i * N)
    var aux_row = (
        bias_p.unsafe_offset(i * N) if epilogue
        == EPILOGUE_RELU_MASK else bias_p
    )
    var b_val = Float32(0.0)
    comptime if epilogue == EPILOGUE_BIAS or epilogue == EPILOGUE_BIAS_RELU:
        b_val = bias_p[unsafe_offset=i]

    var n_vec4_end = (N // SIMD_4W) * SIMD_4W
    var n_vec2_end = (N // SIMD_2W) * SIMD_2W
    var n_vec_end = (N // SIMD_W) * SIMD_W

    var j = 0
    comptime if SIMD_W <= 4:
        while j < n_vec4_end:
            var c0 = SIMD[DType.float32, SIMD_W](0.0)
            var c1 = SIMD[DType.float32, SIMD_W](0.0)
            var c2 = SIMD[DType.float32, SIMD_W](0.0)
            var c3 = SIMD[DType.float32, SIMD_W](0.0)

            for k in range(K):
                var bk = bp.unsafe_offset(k * N + j)
                var bv0 = bk.unsafe_load[width=SIMD_W](0)
                var bv1 = bk.unsafe_load[width=SIMD_W](SIMD_W)
                var bv2 = bk.unsafe_load[width=SIMD_W](2 * SIMD_W)
                var bv3 = bk.unsafe_load[width=SIMD_W](3 * SIMD_W)
                var av = SIMD[DType.float32, SIMD_W](a_row[unsafe_offset=k])
                c0 = av.fma(bv0, c0)
                c1 = av.fma(bv1, c1)
                c2 = av.fma(bv2, c2)
                c3 = av.fma(bv3, c3)

            _store_nn_tile_row[epilogue, SIMD_W](c_row, aux_row, j, c0, b_val)
            _store_nn_tile_row[epilogue, SIMD_W](
                c_row, aux_row, j + SIMD_W, c1, b_val
            )
            _store_nn_tile_row[epilogue, SIMD_W](
                c_row, aux_row, j + 2 * SIMD_W, c2, b_val
            )
            _store_nn_tile_row[epilogue, SIMD_W](
                c_row, aux_row, j + 3 * SIMD_W, c3, b_val
            )
            j += SIMD_4W

    while j < n_vec2_end:
        var c0 = SIMD[DType.float32, SIMD_W](0.0)
        var c1 = SIMD[DType.float32, SIMD_W](0.0)
        for k in range(K):
            var bv0 = bp.unsafe_load[width=SIMD_W](k * N + j)
            var bv1 = bp.unsafe_load[width=SIMD_W](k * N + j + SIMD_W)
            var av = SIMD[DType.float32, SIMD_W](a_row[unsafe_offset=k])
            c0 = av.fma(bv0, c0)
            c1 = av.fma(bv1, c1)
        _store_nn_tile_row[epilogue, SIMD_W](c_row, aux_row, j, c0, b_val)
        _store_nn_tile_row[epilogue, SIMD_W](
            c_row, aux_row, j + SIMD_W, c1, b_val
        )
        j += SIMD_2W

    comptime if N_s < 0 or (N_s % SIMD_2W) >= SIMD_W:
        while j < n_vec_end:
            var c0 = SIMD[DType.float32, SIMD_W](0.0)
            for k in range(K):
                var bv = bp.unsafe_load[width=SIMD_W](k * N + j)
                c0 = SIMD[DType.float32, SIMD_W](a_row[unsafe_offset=k]).fma(
                    bv, c0
                )
            _store_nn_tile_row[epilogue, SIMD_W](c_row, aux_row, j, c0, b_val)
            j += SIMD_W

    comptime if N_s < 0 or (N_s % SIMD_W) != 0:
        while j < N:
            var s = Float32(0.0)
            for k in range(K):
                s = a_row[unsafe_offset=k] * bp[unsafe_offset=k * N + j] + s
            _store_nn_tile_row[epilogue, 1](c_row, aux_row, j, s, b_val)
            j += 1


@inline(.always)
def _matmul_nn_matvec[
    epilogue: Int
](
    ap: ImmPointer[Float32, _],
    bp: ImmPointer[Float32, _],
    cp: MutPointer[Float32, _],
    bias_p: ImmPointer[Float32, _],
    M: Int,
    K: Int,
):
    """Specialized matrix-vector product C[:, 0] = A @ B[:, 0] (+ optional bias/ReLU).
    """
    var k_vec_end = (K // SIMD_W) * SIMD_W
    var m4_end = (M // 4) * 4

    var i = 0
    while i < m4_end:
        var a0p = ap.unsafe_offset(i * K)
        var a1p = ap.unsafe_offset((i + 1) * K)
        var a2p = ap.unsafe_offset((i + 2) * K)
        var a3p = ap.unsafe_offset((i + 3) * K)

        var acc0 = SIMD[DType.float32, SIMD_W](0.0)
        var acc1 = SIMD[DType.float32, SIMD_W](0.0)
        var acc2 = SIMD[DType.float32, SIMD_W](0.0)
        var acc3 = SIMD[DType.float32, SIMD_W](0.0)

        var k = 0
        while k < k_vec_end:
            var bv = bp.unsafe_load[width=SIMD_W](k)
            acc0 = a0p.unsafe_load[width=SIMD_W](k).fma(bv, acc0)
            acc1 = a1p.unsafe_load[width=SIMD_W](k).fma(bv, acc1)
            acc2 = a2p.unsafe_load[width=SIMD_W](k).fma(bv, acc2)
            acc3 = a3p.unsafe_load[width=SIMD_W](k).fma(bv, acc3)
            k += SIMD_W

        var s0 = acc0.reduce_add()
        var s1 = acc1.reduce_add()
        var s2 = acc2.reduce_add()
        var s3 = acc3.reduce_add()

        while k < K:
            var bv = bp[unsafe_offset=k]
            s0 += a0p[unsafe_offset=k] * bv
            s1 += a1p[unsafe_offset=k] * bv
            s2 += a2p[unsafe_offset=k] * bv
            s3 += a3p[unsafe_offset=k] * bv
            k += 1

        comptime if epilogue == EPILOGUE_BIAS:
            s0 += bias_p[unsafe_offset=i]
            s1 += bias_p[unsafe_offset=i + 1]
            s2 += bias_p[unsafe_offset=i + 2]
            s3 += bias_p[unsafe_offset=i + 3]
        elif epilogue == EPILOGUE_BIAS_RELU:
            s0 = max(s0 + bias_p[unsafe_offset=i], Float32(0.0))
            s1 = max(s1 + bias_p[unsafe_offset=i + 1], Float32(0.0))
            s2 = max(s2 + bias_p[unsafe_offset=i + 2], Float32(0.0))
            s3 = max(s3 + bias_p[unsafe_offset=i + 3], Float32(0.0))

        cp[unsafe_offset=i] = s0
        cp[unsafe_offset=i + 1] = s1
        cp[unsafe_offset=i + 2] = s2
        cp[unsafe_offset=i + 3] = s3
        i += 4

    while i < M:
        var a_row = ap.unsafe_offset(i * K)
        var acc = SIMD[DType.float32, SIMD_W](0.0)
        var k = 0
        while k < k_vec_end:
            acc = a_row.unsafe_load[width=SIMD_W](k).fma(
                bp.unsafe_load[width=SIMD_W](k), acc
            )
            k += SIMD_W
        var s = acc.reduce_add()
        while k < K:
            s += a_row[unsafe_offset=k] * bp[unsafe_offset=k]
            k += 1
        comptime if epilogue == EPILOGUE_BIAS:
            s += bias_p[unsafe_offset=i]
        elif epilogue == EPILOGUE_BIAS_RELU:
            s = max(s + bias_p[unsafe_offset=i], Float32(0.0))
        cp[unsafe_offset=i] = s
        i += 1


@inline(.always)
def _matmul_nn_impl[
    M_s: Int = -1, N_s: Int = -1, K_s: Int = -1, epilogue: Int = EPILOGUE_NONE
](
    ap: ImmPointer[Float32, _],
    bp: ImmPointer[Float32, _],
    cp: MutPointer[Float32, _],
    bias_p: ImmPointer[Float32, _],
    m_rt: Int,
    n_rt: Int,
    k_rt: Int,
):
    var M = M_s if M_s > 0 else m_rt
    var N = N_s if N_s > 0 else n_rt
    var K = K_s if K_s > 0 else k_rt

    comptime if N_s == 1:
        _matmul_nn_matvec[epilogue](ap, bp, cp, bias_p, M, K)
        return
    else:
        if N == 1:
            _matmul_nn_matvec[epilogue](ap, bp, cp, bias_p, M, K)
            return

    var num_blocks = M // 4

    comptime if M_s > 0 and N_s > 0 and K_s > 0:
        comptime flops = M_s * N_s * K_s
        comptime if (
            flops >= PARALLEL_FLOP_THRESHOLD
            and (M_s // 4) >= 4
            and (N_s % SIMD_4W == 0)
            and K_s >= 128
            and SIMD_W <= 4
        ):
            # L1-cache-tiled schedule: each task processes BLOCKS_PER_TASK row-blocks
            # with j-outer so each 16-col strip of B stays hot in L1 across all rows.
            comptime BLOCKS_PER_TASK = 4
            comptime num_tasks = ceildiv(M_s // 4, BLOCKS_PER_TASK)

            @inline(.always)
            def tiled_worker(
                t: Int,
            ) {
                imm ap, imm bp, imm cp, imm bias_p, imm N, imm K, imm num_blocks
            }:
                var b0 = t * BLOCKS_PER_TASK
                var b1 = min(b0 + BLOCKS_PER_TASK, num_blocks)
                var j = 0
                while j < N_s:
                    for b in range(b0, b1):
                        _matmul_nn_strip4x4[K_s, epilogue](
                            ap, bp, cp, bias_p, b * 4, j, N, K
                        )
                    j += SIMD_4W

            parallelize(tiled_worker, num_tasks)
        elif flops >= PARALLEL_FLOP_THRESHOLD and (M_s // 4) > 0:

            @inline(.always)
            def block_worker_p(
                b: Int,
            ) {imm ap, imm bp, imm cp, imm bias_p, imm N, imm K}:
                _matmul_nn_row4[M_s, N_s, K_s, epilogue](
                    ap, bp, cp, bias_p, b * 4, N, K
                )

            parallelize(block_worker_p, num_blocks)
        else:
            for b in range(num_blocks):
                _matmul_nn_row4[M_s, N_s, K_s, epilogue](
                    ap, bp, cp, bias_p, b * 4, N, K
                )
    else:

        @inline(.always)
        def block_worker_dyn(
            b: Int,
        ) {imm ap, imm bp, imm cp, imm bias_p, imm N, imm K}:
            _matmul_nn_row4[M_s, N_s, K_s, epilogue](
                ap, bp, cp, bias_p, b * 4, N, K
            )

        if M * N * K >= PARALLEL_FLOP_THRESHOLD and num_blocks > 0:
            parallelize(block_worker_dyn, num_blocks)
        else:
            for b in range(num_blocks):
                block_worker_dyn(b)

    comptime if M_s < 0 or (M_s % 4) != 0:
        for i in range(num_blocks * 4, M):
            _matmul_nn_row1[N_s, epilogue](ap, bp, cp, bias_p, i, N, K)


@inline(.always)
def _matmul_nt_row4[
    K_s: Int, has_scale: Bool, epilogue: Int = EPILOGUE_NONE
](
    ap: ImmPointer[Float32, _],
    bp: ImmPointer[Float32, _],
    cp: MutPointer[Float32, _],
    bias_p: ImmPointer[Float32, _],
    i: Int,
    N: Int,
    K: Int,
    scale: Float32,
):
    """Computes 4 rows [i..i+3] of C = scale * (A @ B.T) (+ optional bias_row) with 4x4 register tiling.
    """
    var a0p = ap.unsafe_offset(i * K)
    var a1p = ap.unsafe_offset((i + 1) * K)
    var a2p = ap.unsafe_offset((i + 2) * K)
    var a3p = ap.unsafe_offset((i + 3) * K)
    var c0p = cp.unsafe_offset(i * N)
    var c1p = cp.unsafe_offset((i + 1) * N)
    var c2p = cp.unsafe_offset((i + 2) * N)
    var c3p = cp.unsafe_offset((i + 3) * N)

    var k_vec_end = (K // SIMD_W) * SIMD_W
    var n4_end = (N // 4) * 4

    var j = 0
    while j < n4_end:
        var b0p = bp.unsafe_offset(j * K)
        var b1p = bp.unsafe_offset((j + 1) * K)
        var b2p = bp.unsafe_offset((j + 2) * K)
        var b3p = bp.unsafe_offset((j + 3) * K)

        var v00 = SIMD[DType.float32, SIMD_W](0.0)
        var v01 = SIMD[DType.float32, SIMD_W](0.0)
        var v02 = SIMD[DType.float32, SIMD_W](0.0)
        var v03 = SIMD[DType.float32, SIMD_W](0.0)
        var v10 = SIMD[DType.float32, SIMD_W](0.0)
        var v11 = SIMD[DType.float32, SIMD_W](0.0)
        var v12 = SIMD[DType.float32, SIMD_W](0.0)
        var v13 = SIMD[DType.float32, SIMD_W](0.0)
        var v20 = SIMD[DType.float32, SIMD_W](0.0)
        var v21 = SIMD[DType.float32, SIMD_W](0.0)
        var v22 = SIMD[DType.float32, SIMD_W](0.0)
        var v23 = SIMD[DType.float32, SIMD_W](0.0)
        var v30 = SIMD[DType.float32, SIMD_W](0.0)
        var v31 = SIMD[DType.float32, SIMD_W](0.0)
        var v32 = SIMD[DType.float32, SIMD_W](0.0)
        var v33 = SIMD[DType.float32, SIMD_W](0.0)

        var k = 0
        while k < k_vec_end:
            var a0 = a0p.unsafe_load[width=SIMD_W](k)
            var a1 = a1p.unsafe_load[width=SIMD_W](k)
            var a2 = a2p.unsafe_load[width=SIMD_W](k)
            var a3 = a3p.unsafe_load[width=SIMD_W](k)
            var b0 = b0p.unsafe_load[width=SIMD_W](k)
            var b1 = b1p.unsafe_load[width=SIMD_W](k)
            var b2 = b2p.unsafe_load[width=SIMD_W](k)
            var b3 = b3p.unsafe_load[width=SIMD_W](k)

            v00 = a0.fma(b0, v00)
            v01 = a0.fma(b1, v01)
            v02 = a0.fma(b2, v02)
            v03 = a0.fma(b3, v03)
            v10 = a1.fma(b0, v10)
            v11 = a1.fma(b1, v11)
            v12 = a1.fma(b2, v12)
            v13 = a1.fma(b3, v13)
            v20 = a2.fma(b0, v20)
            v21 = a2.fma(b1, v21)
            v22 = a2.fma(b2, v22)
            v23 = a2.fma(b3, v23)
            v30 = a3.fma(b0, v30)
            v31 = a3.fma(b1, v31)
            v32 = a3.fma(b2, v32)
            v33 = a3.fma(b3, v33)
            k += SIMD_W

        var s00 = v00.reduce_add()
        var s01 = v01.reduce_add()
        var s02 = v02.reduce_add()
        var s03 = v03.reduce_add()
        var s10 = v10.reduce_add()
        var s11 = v11.reduce_add()
        var s12 = v12.reduce_add()
        var s13 = v13.reduce_add()
        var s20 = v20.reduce_add()
        var s21 = v21.reduce_add()
        var s22 = v22.reduce_add()
        var s23 = v23.reduce_add()
        var s30 = v30.reduce_add()
        var s31 = v31.reduce_add()
        var s32 = v32.reduce_add()
        var s33 = v33.reduce_add()

        comptime if K_s < 0 or (K_s % SIMD_W) != 0:
            while k < K:
                var a0 = a0p[unsafe_offset=k]
                var a1 = a1p[unsafe_offset=k]
                var a2 = a2p[unsafe_offset=k]
                var a3 = a3p[unsafe_offset=k]
                var b0 = b0p[unsafe_offset=k]
                var b1 = b1p[unsafe_offset=k]
                var b2 = b2p[unsafe_offset=k]
                var b3 = b3p[unsafe_offset=k]
                s00 += a0 * b0
                s01 += a0 * b1
                s02 += a0 * b2
                s03 += a0 * b3
                s10 += a1 * b0
                s11 += a1 * b1
                s12 += a1 * b2
                s13 += a1 * b3
                s20 += a2 * b0
                s21 += a2 * b1
                s22 += a2 * b2
                s23 += a2 * b3
                s30 += a3 * b0
                s31 += a3 * b1
                s32 += a3 * b2
                s33 += a3 * b3
                k += 1

        comptime if has_scale:
            s00 *= scale
            s01 *= scale
            s02 *= scale
            s03 *= scale
            s10 *= scale
            s11 *= scale
            s12 *= scale
            s13 *= scale
            s20 *= scale
            s21 *= scale
            s22 *= scale
            s23 *= scale
            s30 *= scale
            s31 *= scale
            s32 *= scale
            s33 *= scale

        comptime if epilogue == EPILOGUE_BIAS_ROW:
            var bv = bias_p.unsafe_load[width=4](j)
            var r0 = SIMD[DType.float32, 4](s00, s01, s02, s03) + bv
            var r1 = SIMD[DType.float32, 4](s10, s11, s12, s13) + bv
            var r2 = SIMD[DType.float32, 4](s20, s21, s22, s23) + bv
            var r3 = SIMD[DType.float32, 4](s30, s31, s32, s33) + bv
            c0p.unsafe_store[width=4](j, r0)
            c1p.unsafe_store[width=4](j, r1)
            c2p.unsafe_store[width=4](j, r2)
            c3p.unsafe_store[width=4](j, r3)
        else:
            c0p[unsafe_offset=j] = s00
            c0p[unsafe_offset=j + 1] = s01
            c0p[unsafe_offset=j + 2] = s02
            c0p[unsafe_offset=j + 3] = s03
            c1p[unsafe_offset=j] = s10
            c1p[unsafe_offset=j + 1] = s11
            c1p[unsafe_offset=j + 2] = s12
            c1p[unsafe_offset=j + 3] = s13
            c2p[unsafe_offset=j] = s20
            c2p[unsafe_offset=j + 1] = s21
            c2p[unsafe_offset=j + 2] = s22
            c2p[unsafe_offset=j + 3] = s23
            c3p[unsafe_offset=j] = s30
            c3p[unsafe_offset=j + 1] = s31
            c3p[unsafe_offset=j + 2] = s32
            c3p[unsafe_offset=j + 3] = s33
        j += 4

    while j < N:
        var b_row = bp.unsafe_offset(j * K)
        var v0 = SIMD[DType.float32, SIMD_W](0.0)
        var v1 = SIMD[DType.float32, SIMD_W](0.0)
        var v2 = SIMD[DType.float32, SIMD_W](0.0)
        var v3 = SIMD[DType.float32, SIMD_W](0.0)
        var k = 0
        while k < k_vec_end:
            var bv = b_row.unsafe_load[width=SIMD_W](k)
            v0 = a0p.unsafe_load[width=SIMD_W](k).fma(bv, v0)
            v1 = a1p.unsafe_load[width=SIMD_W](k).fma(bv, v1)
            v2 = a2p.unsafe_load[width=SIMD_W](k).fma(bv, v2)
            v3 = a3p.unsafe_load[width=SIMD_W](k).fma(bv, v3)
            k += SIMD_W
        var s0 = v0.reduce_add()
        var s1 = v1.reduce_add()
        var s2 = v2.reduce_add()
        var s3 = v3.reduce_add()
        comptime if K_s < 0 or (K_s % SIMD_W) != 0:
            while k < K:
                var bv = b_row[unsafe_offset=k]
                s0 += a0p[unsafe_offset=k] * bv
                s1 += a1p[unsafe_offset=k] * bv
                s2 += a2p[unsafe_offset=k] * bv
                s3 += a3p[unsafe_offset=k] * bv
                k += 1
        comptime if has_scale:
            s0 *= scale
            s1 *= scale
            s2 *= scale
            s3 *= scale
        comptime if epilogue == EPILOGUE_BIAS_ROW:
            var bv_j = bias_p[unsafe_offset=j]
            s0 += bv_j
            s1 += bv_j
            s2 += bv_j
            s3 += bv_j
        c0p[unsafe_offset=j] = s0
        c1p[unsafe_offset=j] = s1
        c2p[unsafe_offset=j] = s2
        c3p[unsafe_offset=j] = s3
        j += 1


@inline(.always)
def _matmul_nt_row1[
    K_s: Int, has_scale: Bool, epilogue: Int = EPILOGUE_NONE
](
    ap: ImmPointer[Float32, _],
    bp: ImmPointer[Float32, _],
    cp: MutPointer[Float32, _],
    bias_p: ImmPointer[Float32, _],
    i: Int,
    N: Int,
    K: Int,
    scale: Float32,
):
    var a_row = ap.unsafe_offset(i * K)
    var c_row = cp.unsafe_offset(i * N)
    var k_vec_end = (K // SIMD_W) * SIMD_W
    for j in range(N):
        var b_row = bp.unsafe_offset(j * K)
        var acc = SIMD[DType.float32, SIMD_W](0.0)
        var k = 0
        while k < k_vec_end:
            acc = a_row.unsafe_load[width=SIMD_W](k).fma(
                b_row.unsafe_load[width=SIMD_W](k), acc
            )
            k += SIMD_W
        var s = acc.reduce_add()
        comptime if K_s < 0 or (K_s % SIMD_W) != 0:
            while k < K:
                s += a_row[unsafe_offset=k] * b_row[unsafe_offset=k]
                k += 1
        comptime if has_scale:
            s *= scale
        comptime if epilogue == EPILOGUE_BIAS_ROW:
            s += bias_p[unsafe_offset=j]
        c_row[unsafe_offset=j] = s


@inline(.always)
def _matmul_nt_impl[
    M_s: Int = -1,
    N_s: Int = -1,
    K_s: Int = -1,
    has_scale: Bool = False,
    epilogue: Int = EPILOGUE_NONE,
](
    ap: ImmPointer[Float32, _],
    bp: ImmPointer[Float32, _],
    cp: MutPointer[Float32, _],
    bias_p: ImmPointer[Float32, _],
    m_rt: Int,
    n_rt: Int,
    k_rt: Int,
    scale: Float32 = 1.0,
):
    var M = M_s if M_s > 0 else m_rt
    var N = N_s if N_s > 0 else n_rt
    var K = K_s if K_s > 0 else k_rt
    var num_blocks = M // 4

    @inline(.always)
    def block_worker(
        b: Int,
    ) {imm ap, imm bp, imm cp, imm bias_p, imm N, imm K, imm scale}:
        _matmul_nt_row4[K_s, has_scale, epilogue](
            ap, bp, cp, bias_p, b * 4, N, K, scale
        )

    comptime if M_s > 0 and N_s > 0 and K_s > 0:
        comptime flops = M_s * N_s * K_s
        comptime if flops >= PARALLEL_FLOP_THRESHOLD and (M_s // 4) > 0:
            parallelize(block_worker, num_blocks)
        else:
            for b in range(num_blocks):
                block_worker(b)
    else:
        if M * N * K >= PARALLEL_FLOP_THRESHOLD and num_blocks > 0:
            parallelize(block_worker, num_blocks)
        else:
            for b in range(num_blocks):
                block_worker(b)

    comptime if M_s < 0 or (M_s % 4) != 0:
        for i in range(num_blocks * 4, M):
            _matmul_nt_row1[K_s, has_scale, epilogue](
                ap, bp, cp, bias_p, i, N, K, scale
            )


@inline(.always)
def _store_tn_tile_row[
    has_relu_mask: Bool, has_scale: Bool, w: Int
](
    c_row: MutPointer[Float32, _],
    mask_row: ImmPointer[Float32, _],
    j: Int,
    acc: SIMD[DType.float32, w],
    scale: Float32 = 1.0,
):
    var res = acc
    comptime if has_scale:
        res = res * scale
    comptime if has_relu_mask:
        var m = mask_row.unsafe_load[width=w](j)
        res = m.gt(0.0).select(res, SIMD[DType.float32, w](0.0))
    c_row.unsafe_store[width=w](j, res)


@inline(.always)
def _matmul_tn_strip4x4[
    K_s: Int, has_relu_mask: Bool, has_scale: Bool
](
    ap: ImmPointer[Float32, _],
    bp: ImmPointer[Float32, _],
    mask_p: ImmPointer[Float32, _],
    cp: MutPointer[Float32, _],
    i: Int,
    j: Int,
    M: Int,
    N: Int,
    K: Int,
    scale: Float32,
):
    """Computes a 4 x SIMD_4W tile [i..i+3, j..j+SIMD_4W-1] of C = A.T @ B."""
    var c0p = cp.unsafe_offset(i * N)
    var c1p = cp.unsafe_offset((i + 1) * N)
    var c2p = cp.unsafe_offset((i + 2) * N)
    var c3p = cp.unsafe_offset((i + 3) * N)
    var m0p = mask_p.unsafe_offset(i * N)
    var m1p = mask_p.unsafe_offset((i + 1) * N)
    var m2p = mask_p.unsafe_offset((i + 2) * N)
    var m3p = mask_p.unsafe_offset((i + 3) * N)

    var c00 = SIMD[DType.float32, SIMD_W](0.0)
    var c01 = SIMD[DType.float32, SIMD_W](0.0)
    var c02 = SIMD[DType.float32, SIMD_W](0.0)
    var c03 = SIMD[DType.float32, SIMD_W](0.0)
    var c10 = SIMD[DType.float32, SIMD_W](0.0)
    var c11 = SIMD[DType.float32, SIMD_W](0.0)
    var c12 = SIMD[DType.float32, SIMD_W](0.0)
    var c13 = SIMD[DType.float32, SIMD_W](0.0)
    var c20 = SIMD[DType.float32, SIMD_W](0.0)
    var c21 = SIMD[DType.float32, SIMD_W](0.0)
    var c22 = SIMD[DType.float32, SIMD_W](0.0)
    var c23 = SIMD[DType.float32, SIMD_W](0.0)
    var c30 = SIMD[DType.float32, SIMD_W](0.0)
    var c31 = SIMD[DType.float32, SIMD_W](0.0)
    var c32 = SIMD[DType.float32, SIMD_W](0.0)
    var c33 = SIMD[DType.float32, SIMD_W](0.0)

    comptime if K_s > 0 and K_s <= 16:
        comptime for k in range(K_s):
            var bk = bp.unsafe_offset(k * N + j)
            var bv0 = bk.unsafe_load[width=SIMD_W](0)
            var bv1 = bk.unsafe_load[width=SIMD_W](SIMD_W)
            var bv2 = bk.unsafe_load[width=SIMD_W](2 * SIMD_W)
            var bv3 = bk.unsafe_load[width=SIMD_W](3 * SIMD_W)
            var av = ap.unsafe_load[width=4](k * M + i)
            var a0 = SIMD[DType.float32, SIMD_W](av[0])
            var a1 = SIMD[DType.float32, SIMD_W](av[1])
            var a2 = SIMD[DType.float32, SIMD_W](av[2])
            var a3 = SIMD[DType.float32, SIMD_W](av[3])
            c00 = a0.fma(bv0, c00)
            c01 = a0.fma(bv1, c01)
            c02 = a0.fma(bv2, c02)
            c03 = a0.fma(bv3, c03)
            c10 = a1.fma(bv0, c10)
            c11 = a1.fma(bv1, c11)
            c12 = a1.fma(bv2, c12)
            c13 = a1.fma(bv3, c13)
            c20 = a2.fma(bv0, c20)
            c21 = a2.fma(bv1, c21)
            c22 = a2.fma(bv2, c22)
            c23 = a2.fma(bv3, c23)
            c30 = a3.fma(bv0, c30)
            c31 = a3.fma(bv1, c31)
            c32 = a3.fma(bv2, c32)
            c33 = a3.fma(bv3, c33)
    else:
        for k in range(K):
            var bk = bp.unsafe_offset(k * N + j)
            var bv0 = bk.unsafe_load[width=SIMD_W](0)
            var bv1 = bk.unsafe_load[width=SIMD_W](SIMD_W)
            var bv2 = bk.unsafe_load[width=SIMD_W](2 * SIMD_W)
            var bv3 = bk.unsafe_load[width=SIMD_W](3 * SIMD_W)
            var av = ap.unsafe_load[width=4](k * M + i)
            var a0 = SIMD[DType.float32, SIMD_W](av[0])
            var a1 = SIMD[DType.float32, SIMD_W](av[1])
            var a2 = SIMD[DType.float32, SIMD_W](av[2])
            var a3 = SIMD[DType.float32, SIMD_W](av[3])
            c00 = a0.fma(bv0, c00)
            c01 = a0.fma(bv1, c01)
            c02 = a0.fma(bv2, c02)
            c03 = a0.fma(bv3, c03)
            c10 = a1.fma(bv0, c10)
            c11 = a1.fma(bv1, c11)
            c12 = a1.fma(bv2, c12)
            c13 = a1.fma(bv3, c13)
            c20 = a2.fma(bv0, c20)
            c21 = a2.fma(bv1, c21)
            c22 = a2.fma(bv2, c22)
            c23 = a2.fma(bv3, c23)
            c30 = a3.fma(bv0, c30)
            c31 = a3.fma(bv1, c31)
            c32 = a3.fma(bv2, c32)
            c33 = a3.fma(bv3, c33)

    _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
        c0p, m0p, j, c00, scale
    )
    _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
        c0p, m0p, j + SIMD_W, c01, scale
    )
    _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
        c0p, m0p, j + 2 * SIMD_W, c02, scale
    )
    _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
        c0p, m0p, j + 3 * SIMD_W, c03, scale
    )
    _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
        c1p, m1p, j, c10, scale
    )
    _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
        c1p, m1p, j + SIMD_W, c11, scale
    )
    _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
        c1p, m1p, j + 2 * SIMD_W, c12, scale
    )
    _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
        c1p, m1p, j + 3 * SIMD_W, c13, scale
    )
    _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
        c2p, m2p, j, c20, scale
    )
    _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
        c2p, m2p, j + SIMD_W, c21, scale
    )
    _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
        c2p, m2p, j + 2 * SIMD_W, c22, scale
    )
    _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
        c2p, m2p, j + 3 * SIMD_W, c23, scale
    )
    _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
        c3p, m3p, j, c30, scale
    )
    _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
        c3p, m3p, j + SIMD_W, c31, scale
    )
    _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
        c3p, m3p, j + 2 * SIMD_W, c32, scale
    )
    _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
        c3p, m3p, j + 3 * SIMD_W, c33, scale
    )


@inline(.always)
def _matmul_tn_row4[
    N_s: Int, K_s: Int, has_relu_mask: Bool = False, has_scale: Bool = False
](
    ap: ImmPointer[Float32, _],
    bp: ImmPointer[Float32, _],
    mask_p: ImmPointer[Float32, _],
    cp: MutPointer[Float32, _],
    i: Int,
    M: Int,
    N: Int,
    K: Int,
    scale: Float32 = 1.0,
):
    """Computes 4 output rows [i..i+3] of C = A.T @ B (+ optional ReLU mask / scale) in registers.
    """
    var c0p = cp.unsafe_offset(i * N)
    var c1p = cp.unsafe_offset((i + 1) * N)
    var c2p = cp.unsafe_offset((i + 2) * N)
    var c3p = cp.unsafe_offset((i + 3) * N)
    var m0p = mask_p.unsafe_offset(i * N)
    var m1p = mask_p.unsafe_offset((i + 1) * N)
    var m2p = mask_p.unsafe_offset((i + 2) * N)
    var m3p = mask_p.unsafe_offset((i + 3) * N)

    var n_vec4_end = (N // SIMD_4W) * SIMD_4W
    var n_vec2_end = (N // SIMD_2W) * SIMD_2W
    var n_vec_end = (N // SIMD_W) * SIMD_W

    var j = 0
    comptime if SIMD_W <= 4:
        while j < n_vec4_end:
            _matmul_tn_strip4x4[K_s, has_relu_mask, has_scale](
                ap, bp, mask_p, cp, i, j, M, N, K, scale
            )
            j += SIMD_4W

    while j < n_vec2_end:
        var c00 = SIMD[DType.float32, SIMD_W](0.0)
        var c01 = SIMD[DType.float32, SIMD_W](0.0)
        var c10 = SIMD[DType.float32, SIMD_W](0.0)
        var c11 = SIMD[DType.float32, SIMD_W](0.0)
        var c20 = SIMD[DType.float32, SIMD_W](0.0)
        var c21 = SIMD[DType.float32, SIMD_W](0.0)
        var c30 = SIMD[DType.float32, SIMD_W](0.0)
        var c31 = SIMD[DType.float32, SIMD_W](0.0)

        comptime if K_s > 0 and K_s <= 16:
            comptime for k in range(K_s):
                var bv0 = bp.unsafe_load[width=SIMD_W](k * N + j)
                var bv1 = bp.unsafe_load[width=SIMD_W](k * N + j + SIMD_W)
                var av = ap.unsafe_load[width=4](k * M + i)
                var a0 = SIMD[DType.float32, SIMD_W](av[0])
                var a1 = SIMD[DType.float32, SIMD_W](av[1])
                var a2 = SIMD[DType.float32, SIMD_W](av[2])
                var a3 = SIMD[DType.float32, SIMD_W](av[3])
                c00 = a0.fma(bv0, c00)
                c01 = a0.fma(bv1, c01)
                c10 = a1.fma(bv0, c10)
                c11 = a1.fma(bv1, c11)
                c20 = a2.fma(bv0, c20)
                c21 = a2.fma(bv1, c21)
                c30 = a3.fma(bv0, c30)
                c31 = a3.fma(bv1, c31)
        else:
            for k in range(K):
                var bv0 = bp.unsafe_load[width=SIMD_W](k * N + j)
                var bv1 = bp.unsafe_load[width=SIMD_W](k * N + j + SIMD_W)
                var av = ap.unsafe_load[width=4](k * M + i)
                var a0 = SIMD[DType.float32, SIMD_W](av[0])
                var a1 = SIMD[DType.float32, SIMD_W](av[1])
                var a2 = SIMD[DType.float32, SIMD_W](av[2])
                var a3 = SIMD[DType.float32, SIMD_W](av[3])
                c00 = a0.fma(bv0, c00)
                c01 = a0.fma(bv1, c01)
                c10 = a1.fma(bv0, c10)
                c11 = a1.fma(bv1, c11)
                c20 = a2.fma(bv0, c20)
                c21 = a2.fma(bv1, c21)
                c30 = a3.fma(bv0, c30)
                c31 = a3.fma(bv1, c31)

        _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
            c0p, m0p, j, c00, scale
        )
        _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
            c0p, m0p, j + SIMD_W, c01, scale
        )
        _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
            c1p, m1p, j, c10, scale
        )
        _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
            c1p, m1p, j + SIMD_W, c11, scale
        )
        _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
            c2p, m2p, j, c20, scale
        )
        _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
            c2p, m2p, j + SIMD_W, c21, scale
        )
        _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
            c3p, m3p, j, c30, scale
        )
        _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
            c3p, m3p, j + SIMD_W, c31, scale
        )
        j += SIMD_2W

    comptime if N_s < 0 or (N_s % SIMD_2W) >= SIMD_W:
        while j < n_vec_end:
            var c0 = SIMD[DType.float32, SIMD_W](0.0)
            var c1 = SIMD[DType.float32, SIMD_W](0.0)
            var c2 = SIMD[DType.float32, SIMD_W](0.0)
            var c3 = SIMD[DType.float32, SIMD_W](0.0)
            for k in range(K):
                var bv = bp.unsafe_load[width=SIMD_W](k * N + j)
                var av = ap.unsafe_load[width=4](k * M + i)
                c0 = SIMD[DType.float32, SIMD_W](av[0]).fma(bv, c0)
                c1 = SIMD[DType.float32, SIMD_W](av[1]).fma(bv, c1)
                c2 = SIMD[DType.float32, SIMD_W](av[2]).fma(bv, c2)
                c3 = SIMD[DType.float32, SIMD_W](av[3]).fma(bv, c3)
            _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
                c0p, m0p, j, c0, scale
            )
            _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
                c1p, m1p, j, c1, scale
            )
            _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
                c2p, m2p, j, c2, scale
            )
            _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
                c3p, m3p, j, c3, scale
            )
            j += SIMD_W

    comptime if N_s < 0 or (N_s % SIMD_W) != 0:
        while j < N:
            var s0 = Float32(0.0)
            var s1 = Float32(0.0)
            var s2 = Float32(0.0)
            var s3 = Float32(0.0)
            for k in range(K):
                var bv = bp[unsafe_offset=k * N + j]
                var av = ap.unsafe_load[width=4](k * M + i)
                s0 += av[0] * bv
                s1 += av[1] * bv
                s2 += av[2] * bv
                s3 += av[3] * bv
            _store_tn_tile_row[has_relu_mask, has_scale, 1](
                c0p, m0p, j, s0, scale
            )
            _store_tn_tile_row[has_relu_mask, has_scale, 1](
                c1p, m1p, j, s1, scale
            )
            _store_tn_tile_row[has_relu_mask, has_scale, 1](
                c2p, m2p, j, s2, scale
            )
            _store_tn_tile_row[has_relu_mask, has_scale, 1](
                c3p, m3p, j, s3, scale
            )
            j += 1


@inline(.always)
def _matmul_tn_row1[
    has_relu_mask: Bool = False, has_scale: Bool = False
](
    ap: ImmPointer[Float32, _],
    bp: ImmPointer[Float32, _],
    mask_p: ImmPointer[Float32, _],
    cp: MutPointer[Float32, _],
    i: Int,
    M: Int,
    N: Int,
    K: Int,
    scale: Float32 = 1.0,
):
    var c_row = cp.unsafe_offset(i * N)
    var m_row = mask_p.unsafe_offset(i * N)
    var n_vec_end = (N // SIMD_W) * SIMD_W
    var j = 0
    while j < n_vec_end:
        var c0 = SIMD[DType.float32, SIMD_W](0.0)
        for k in range(K):
            var bv = bp.unsafe_load[width=SIMD_W](k * N + j)
            c0 = SIMD[DType.float32, SIMD_W](ap[unsafe_offset=k * M + i]).fma(
                bv, c0
            )
        _store_tn_tile_row[has_relu_mask, has_scale, SIMD_W](
            c_row, m_row, j, c0, scale
        )
        j += SIMD_W
    while j < N:
        var s = Float32(0.0)
        for k in range(K):
            s += ap[unsafe_offset=k * M + i] * bp[unsafe_offset=k * N + j]
        _store_tn_tile_row[has_relu_mask, has_scale, 1](
            c_row, m_row, j, s, scale
        )
        j += 1


@inline(.always)
def _matmul_tn_impl[
    M_s: Int = -1,
    N_s: Int = -1,
    K_s: Int = -1,
    has_relu_mask: Bool = False,
    has_scale: Bool = False,
](
    ap: ImmPointer[Float32, _],
    bp: ImmPointer[Float32, _],
    mask_p: ImmPointer[Float32, _],
    cp: MutPointer[Float32, _],
    m_rt: Int,
    n_rt: Int,
    k_rt: Int,
    scale: Float32 = 1.0,
):
    var M = M_s if M_s > 0 else m_rt
    var N = N_s if N_s > 0 else n_rt
    var K = K_s if K_s > 0 else k_rt
    var num_blocks = M // 4

    comptime if M_s > 0 and N_s > 0 and K_s > 0:
        comptime flops = M_s * N_s * K_s
        comptime if (
            flops >= PARALLEL_FLOP_THRESHOLD
            and (M_s // 4) >= 8
            and (N_s % SIMD_4W == 0)
            and SIMD_W <= 4
        ):
            # L1-cache-tiled schedule: each task processes 8 row-blocks (32 rows)
            # with j-outer so each 16-col strip of B stays hot in L1 cache.
            comptime BLOCKS_PER_TASK = 8
            comptime num_tasks = ceildiv(M_s // 4, BLOCKS_PER_TASK)

            @inline(.always)
            def tiled_tn_worker(
                t: Int,
            ) {
                imm ap,
                imm bp,
                imm mask_p,
                imm cp,
                imm M,
                imm N,
                imm K,
                imm scale,
                imm num_blocks,
            }:
                var b0 = t * BLOCKS_PER_TASK
                var b1 = min(b0 + BLOCKS_PER_TASK, num_blocks)
                var j = 0
                while j < N_s:
                    for b in range(b0, b1):
                        _matmul_tn_strip4x4[K_s, has_relu_mask, has_scale](
                            ap, bp, mask_p, cp, b * 4, j, M, N, K, scale
                        )
                    j += SIMD_4W

            parallelize(tiled_tn_worker, num_tasks)
        elif flops >= PARALLEL_FLOP_THRESHOLD and (M_s // 4) > 0:

            @inline(.always)
            def block_tn_worker_p(
                b: Int,
            ) {
                imm ap,
                imm bp,
                imm mask_p,
                imm cp,
                imm M,
                imm N,
                imm K,
                imm scale,
            }:
                _matmul_tn_row4[N_s, K_s, has_relu_mask, has_scale](
                    ap, bp, mask_p, cp, b * 4, M, N, K, scale
                )

            parallelize(block_tn_worker_p, num_blocks)
        else:
            for b in range(num_blocks):
                _matmul_tn_row4[N_s, K_s, has_relu_mask, has_scale](
                    ap, bp, mask_p, cp, b * 4, M, N, K, scale
                )
    else:

        @inline(.always)
        def block_tn_worker_dyn(
            b: Int,
        ) {imm ap, imm bp, imm mask_p, imm cp, imm M, imm N, imm K, imm scale}:
            _matmul_tn_row4[N_s, K_s, has_relu_mask, has_scale](
                ap, bp, mask_p, cp, b * 4, M, N, K, scale
            )

        if M * N * K >= PARALLEL_FLOP_THRESHOLD and num_blocks > 0:
            parallelize(block_tn_worker_dyn, num_blocks)
        else:
            for b in range(num_blocks):
                block_tn_worker_dyn(b)

    comptime if M_s < 0 or (M_s % 4) != 0:
        for i in range(num_blocks * 4, M):
            _matmul_tn_row1[has_relu_mask, has_scale](
                ap, bp, mask_p, cp, i, M, N, K, scale
            )


def cpu_gather_batch_cols(
    pixels: Span[Float32, _],
    indices: Span[Int, _],
    batch_start: Int,
    batch_size: Int,
    num_features: Int,
    mut out_x: CPUMatrix,
    mut out_labels: List[Int],
    all_labels: Span[Int, _],
):
    """Gathers a mini-batch of samples into column-major sample layout (features, batch_size).
    """
    var src_ptr = pixels.unsafe_ptr()
    var idx_ptr = indices.unsafe_ptr()
    var x_ptr = out_x.data.unsafe_ptr()

    for b in range(batch_size):
        out_labels[b] = all_labels[idx_ptr[unsafe_offset=batch_start + b]]

    # Chunk across workers so each task writes multiple columns and avoids excessive dispatch
    comptime COLS_PER_TASK = 8
    var num_tasks = ceildiv(batch_size, COLS_PER_TASK)

    @inline(.always)
    def worker(
        t: Int,
    ) {
        imm src_ptr,
        imm idx_ptr,
        imm x_ptr,
        imm batch_start,
        imm batch_size,
        imm num_features,
    }:
        var b0 = t * COLS_PER_TASK
        var b_end = min(b0 + COLS_PER_TASK, batch_size)
        if b_end - b0 == 4:
            var s0 = idx_ptr[unsafe_offset=batch_start + b0] * num_features
            var s1 = idx_ptr[unsafe_offset=batch_start + b0 + 1] * num_features
            var s2 = idx_ptr[unsafe_offset=batch_start + b0 + 2] * num_features
            var s3 = idx_ptr[unsafe_offset=batch_start + b0 + 3] * num_features
            for f in range(num_features):
                var dst = x_ptr.unsafe_offset(f * batch_size + b0)
                var v = SIMD[DType.float32, 4](
                    src_ptr[unsafe_offset=s0 + f],
                    src_ptr[unsafe_offset=s1 + f],
                    src_ptr[unsafe_offset=s2 + f],
                    src_ptr[unsafe_offset=s3 + f],
                )
                dst.unsafe_store[width=4](0, v)
        elif b_end - b0 == 8:
            var s0 = idx_ptr[unsafe_offset=batch_start + b0] * num_features
            var s1 = idx_ptr[unsafe_offset=batch_start + b0 + 1] * num_features
            var s2 = idx_ptr[unsafe_offset=batch_start + b0 + 2] * num_features
            var s3 = idx_ptr[unsafe_offset=batch_start + b0 + 3] * num_features
            var s4 = idx_ptr[unsafe_offset=batch_start + b0 + 4] * num_features
            var s5 = idx_ptr[unsafe_offset=batch_start + b0 + 5] * num_features
            var s6 = idx_ptr[unsafe_offset=batch_start + b0 + 6] * num_features
            var s7 = idx_ptr[unsafe_offset=batch_start + b0 + 7] * num_features
            for f in range(num_features):
                var dst = x_ptr.unsafe_offset(f * batch_size + b0)
                var v0 = SIMD[DType.float32, 4](
                    src_ptr[unsafe_offset=s0 + f],
                    src_ptr[unsafe_offset=s1 + f],
                    src_ptr[unsafe_offset=s2 + f],
                    src_ptr[unsafe_offset=s3 + f],
                )
                var v1 = SIMD[DType.float32, 4](
                    src_ptr[unsafe_offset=s4 + f],
                    src_ptr[unsafe_offset=s5 + f],
                    src_ptr[unsafe_offset=s6 + f],
                    src_ptr[unsafe_offset=s7 + f],
                )
                dst.unsafe_store[width=4](0, v0)
                dst.unsafe_store[width=4](4, v1)
        else:
            for b in range(b0, b_end):
                var sample_base = (
                    idx_ptr[unsafe_offset=batch_start + b] * num_features
                )
                for f in range(num_features):
                    x_ptr[unsafe_offset=f * batch_size + b] = src_ptr[
                        unsafe_offset=sample_base + f
                    ]

    parallelize(worker, num_tasks)


def cpu_gather_eval_cols(
    pixels: Span[Float32, _],
    start_idx: Int,
    num_samples: Int,
    num_features: Int,
    mut out_x: CPUMatrix,
):
    """Copies contiguous samples [start_idx, start_idx + num_samples) into (num_features, num_samples).
    """
    var src_ptr = pixels.unsafe_ptr()
    var x_ptr = out_x.data.unsafe_ptr()
    comptime COLS_PER_TASK = 8
    var num_tasks = ceildiv(num_samples, COLS_PER_TASK)

    @inline(.always)
    def worker(
        t: Int,
    ) {
        imm src_ptr, imm x_ptr, imm start_idx, imm num_samples, imm num_features
    }:
        var s0 = t * COLS_PER_TASK
        var s_end = min(s0 + COLS_PER_TASK, num_samples)
        for s in range(s0, s_end):
            var sample_base = (start_idx + s) * num_features
            for f in range(num_features):
                x_ptr[unsafe_offset=f * num_samples + s] = src_ptr[
                    unsafe_offset=sample_base + f
                ]

    parallelize(worker, num_tasks)


def cpu_gather_batch_rows(
    pixels: Span[Float32, _],
    indices: Span[Int, _],
    batch_start: Int,
    batch_size: Int,
    num_features: Int,
    mut out_x: CPUMatrix,
    mut out_labels: List[Int],
    all_labels: Span[Int, _],
):
    """Gathers a mini-batch of samples into row-major sample layout (batch_size, features).
    """
    var src_ptr = pixels.unsafe_ptr()
    var idx_ptr = indices.unsafe_ptr()
    var x_ptr = out_x.data.unsafe_ptr()

    for b in range(batch_size):
        out_labels[b] = all_labels[idx_ptr[unsafe_offset=batch_start + b]]

    # 4 worker tasks copying contiguous num_features rows sequentially
    comptime ROWS_PER_TASK = 25
    var num_tasks = ceildiv(batch_size, ROWS_PER_TASK)

    @inline(.always)
    def row_copy_worker(
        t: Int,
    ) {
        imm src_ptr,
        imm idx_ptr,
        imm x_ptr,
        imm batch_start,
        imm batch_size,
        imm num_features,
    }:
        var b0 = t * ROWS_PER_TASK
        var b_end = min(b0 + ROWS_PER_TASK, batch_size)
        for b in range(b0, b_end):
            var sample_idx = idx_ptr[unsafe_offset=batch_start + b]
            unsafe_memcpy(
                dest=x_ptr.unsafe_offset(b * num_features),
                src=src_ptr.unsafe_offset(sample_idx * num_features),
                count=num_features,
            )

    parallelize(row_copy_worker, num_tasks)


@inline(.always)
def cpu_transpose_static[
    rows: Int, cols: Int
](
    src: StaticMatrixView[rows, cols, _],
    dst: StaticMatrixView[mut=True, cols, rows, _],
):
    """Transposes a (rows, cols) matrix into (cols, rows)."""
    var sp = src.ptr
    var dp = dst.ptr
    for r in range(rows):
        var sr = sp.unsafe_offset(r * cols)
        for c in range(cols):
            dp[unsafe_offset=c * rows + r] = sr[unsafe_offset=c]


@inline(.always)
def cpu_matmul_nn(a: CPUMatrix, b: CPUMatrix, mut c: CPUMatrix) raises:
    """Computes C = A @ B where A:(M, K), B:(K, N), C:(M, N)."""
    if a.cols != b.rows or c.rows != a.rows or c.cols != b.cols:
        raise Error("cpu_matmul_nn: shape mismatch")
    _matmul_nn_impl(
        a.data.unsafe_ptr(),
        b.data.unsafe_ptr(),
        c.data.unsafe_ptr(),
        a.data.unsafe_ptr(),
        a.rows,
        b.cols,
        a.cols,
    )


@inline(.always)
def cpu_matmul_nn_static[
    M: Int, N: Int, K: Int
](
    a: StaticMatrixView[M, K, _],
    b: StaticMatrixView[K, N, _],
    c: StaticMatrixView[mut=True, M, N, _],
):
    """Compile-time specialized C = A @ B."""
    _matmul_nn_impl[M, N, K, EPILOGUE_NONE](a.ptr, b.ptr, c.ptr, a.ptr, M, N, K)


@inline(.always)
def cpu_matmul_nn_static[
    M: Int, N: Int, K: Int, epilogue: Int = EPILOGUE_NONE
](
    a: StaticMatrixView[M, K, _],
    b: StaticMatrixView[K, N, _],
    c: StaticMatrixView[mut=True, M, N, _],
    bias: StaticMatrixView[M, 1, _],
):
    """Compile-time specialized C = A @ B (+ optional col bias / ReLU)."""
    _matmul_nn_impl[M, N, K, epilogue](a.ptr, b.ptr, c.ptr, bias.ptr, M, N, K)


@inline(.always)
def cpu_matmul_nn_bias_row_static[
    M: Int, N: Int, K: Int
](
    a: StaticMatrixView[M, K, _],
    b: StaticMatrixView[K, N, _],
    c: StaticMatrixView[mut=True, M, N, _],
    bias: StaticMatrixView[N, 1, _],
):
    """Compile-time specialized C = A @ B + row_bias."""
    _matmul_nn_impl[M, N, K, EPILOGUE_BIAS_ROW](
        a.ptr, b.ptr, c.ptr, bias.ptr, M, N, K
    )


@inline(.always)
def cpu_matmul_nn_bias_row_relu_static[
    M: Int, N: Int, K: Int
](
    a: StaticMatrixView[M, K, _],
    b: StaticMatrixView[K, N, _],
    c: StaticMatrixView[mut=True, M, N, _],
    bias: StaticMatrixView[N, 1, _],
):
    """Compile-time specialized C = ReLU(A @ B + row_bias)."""
    _matmul_nn_impl[M, N, K, EPILOGUE_BIAS_ROW_RELU](
        a.ptr, b.ptr, c.ptr, bias.ptr, M, N, K
    )


@inline(.always)
def cpu_matmul_nn_relu_mask_static[
    M: Int, N: Int, K: Int
](
    a: StaticMatrixView[M, K, _],
    b: StaticMatrixView[K, N, _],
    mask_act: StaticMatrixView[M, N, _],
    c: StaticMatrixView[mut=True, M, N, _],
):
    """Compile-time specialized C = (A @ B) * (mask_act > 0)."""
    _matmul_nn_impl[M, N, K, EPILOGUE_RELU_MASK](
        a.ptr, b.ptr, c.ptr, mask_act.ptr, M, N, K
    )


@inline(.always)
def cpu_matmul_nt(
    a: CPUMatrix, b: CPUMatrix, mut c: CPUMatrix, scale: Float32 = 1.0
) raises:
    """Computes C = scale * (A @ B.T) where A:(M, K), B:(N, K), C:(M, N)."""
    if a.cols != b.cols or c.rows != a.rows or c.cols != b.rows:
        raise Error("cpu_matmul_nt: shape mismatch")
    var ap = a.data.unsafe_ptr()
    if scale == 1.0:
        _matmul_nt_impl[has_scale=False](
            ap,
            b.data.unsafe_ptr(),
            c.data.unsafe_ptr(),
            ap,
            a.rows,
            b.rows,
            a.cols,
        )
    else:
        _matmul_nt_impl[has_scale=True](
            ap,
            b.data.unsafe_ptr(),
            c.data.unsafe_ptr(),
            ap,
            a.rows,
            b.rows,
            a.cols,
            scale,
        )


@inline(.always)
def cpu_matmul_nt_static[
    M: Int, N: Int, K: Int, has_scale: Bool = False
](
    a: StaticMatrixView[M, K, _],
    b: StaticMatrixView[N, K, _],
    c: StaticMatrixView[mut=True, M, N, _],
    scale: Float32 = 1.0,
):
    """Compile-time specialized C = scale * (A @ B.T)."""
    _matmul_nt_impl[M, N, K, has_scale](
        a.ptr, b.ptr, c.ptr, a.ptr, M, N, K, scale
    )


@inline(.always)
def cpu_matmul_nt_bias_row_static[
    M: Int, N: Int, K: Int
](
    a: StaticMatrixView[M, K, _],
    b: StaticMatrixView[N, K, _],
    c: StaticMatrixView[mut=True, M, N, _],
    bias: StaticMatrixView[N, 1, _],
):
    """Compile-time specialized C = A @ B.T + row_bias."""
    _matmul_nt_impl[M, N, K, False, EPILOGUE_BIAS_ROW](
        a.ptr, b.ptr, c.ptr, bias.ptr, M, N, K, 1.0
    )


@inline(.always)
def cpu_matmul_tn(a: CPUMatrix, b: CPUMatrix, mut c: CPUMatrix) raises:
    """Computes C = A.T @ B where A:(K, M), B:(K, N), C:(M, N)."""
    if a.rows != b.rows or c.rows != a.cols or c.cols != b.cols:
        raise Error("cpu_matmul_tn: shape mismatch")
    _matmul_tn_impl(
        a.data.unsafe_ptr(),
        b.data.unsafe_ptr(),
        a.data.unsafe_ptr(),
        c.data.unsafe_ptr(),
        a.cols,
        b.cols,
        a.rows,
    )


@inline(.always)
def cpu_matmul_tn_static[
    M: Int, N: Int, K: Int
](
    a: StaticMatrixView[K, M, _],
    b: StaticMatrixView[K, N, _],
    c: StaticMatrixView[mut=True, M, N, _],
):
    """Compile-time specialized C = A.T @ B."""
    _matmul_tn_impl[M, N, K, False, False](a.ptr, b.ptr, a.ptr, c.ptr, M, N, K)


@inline(.always)
def cpu_matmul_tn_scaled_static[
    M: Int, N: Int, K: Int
](
    a: StaticMatrixView[K, M, _],
    b: StaticMatrixView[K, N, _],
    c: StaticMatrixView[mut=True, M, N, _],
    scale: Float32,
):
    """Compile-time specialized C = scale * (A.T @ B)."""
    _matmul_tn_impl[M, N, K, False, True](
        a.ptr, b.ptr, a.ptr, c.ptr, M, N, K, scale
    )


@inline(.always)
def cpu_matmul_tn_relu_mask_static[
    M: Int, N: Int, K: Int
](
    a: StaticMatrixView[K, M, _],
    b: StaticMatrixView[K, N, _],
    mask_act: StaticMatrixView[M, N, _],
    c: StaticMatrixView[mut=True, M, N, _],
):
    """Compile-time specialized C = (A.T @ B) * (mask_act > 0)."""
    _matmul_tn_impl[M, N, K, True, False](
        a.ptr, b.ptr, mask_act.ptr, c.ptr, M, N, K
    )


def cpu_add_bias_and_relu(
    mut z: CPUMatrix, bias: CPUMatrix, mut a: CPUMatrix
) raises:
    """Computes Z[r, c] += bias[r] and A[r, c] = max(Z[r, c], 0)."""
    if (
        z.rows != bias.rows
        or bias.cols != 1
        or a.rows != z.rows
        or a.cols != z.cols
    ):
        raise Error("cpu_add_bias_and_relu: shape mismatch")
    var rows = z.rows
    var cols = z.cols
    var zp = z.data.unsafe_ptr()
    var bp = bias.data.unsafe_ptr()
    var ap = a.data.unsafe_ptr()

    @inline(.always)
    def row_worker(r: Int) {imm zp, imm bp, imm ap, imm cols}:
        var bv = bp[unsafe_offset=r]
        var z_row = zp.unsafe_offset(r * cols)
        var a_row = ap.unsafe_offset(r * cols)

        @inline(.always)
        def _op[w: Int](c: Int) {imm z_row, imm a_row, imm bv}:
            var zv = z_row.unsafe_load[width=w](c) + bv
            z_row.unsafe_store[width=w](c, zv)
            a_row.unsafe_store[width=w](c, max(zv, SIMD[DType.float32, w](0.0)))

        vectorize[SIMD_W](cols, _op)

    if rows * cols >= PARALLEL_ELEM_THRESHOLD:
        parallelize(row_worker, rows)
    else:
        for r in range(rows):
            row_worker(r)


def cpu_add_bias(mut z: CPUMatrix, bias: CPUMatrix) raises:
    """Computes Z[r, c] += bias[r] in-place."""
    if z.rows != bias.rows or bias.cols != 1:
        raise Error("cpu_add_bias: shape mismatch")
    var rows = z.rows
    var cols = z.cols
    var zp = z.data.unsafe_ptr()
    var bp = bias.data.unsafe_ptr()

    @inline(.always)
    def row_worker(r: Int) {imm zp, imm bp, imm cols}:
        var bv = bp[unsafe_offset=r]
        var z_row = zp.unsafe_offset(r * cols)

        @inline(.always)
        def _op[w: Int](c: Int) {imm z_row, imm bv}:
            var zv = z_row.unsafe_load[width=w](c) + bv
            z_row.unsafe_store[width=w](c, zv)

        vectorize[SIMD_W](cols, _op)

    if rows * cols >= PARALLEL_ELEM_THRESHOLD:
        parallelize(row_worker, rows)
    else:
        for r in range(rows):
            row_worker(r)


@inline(.always)
def _softmax_forward_simd_cols[
    rows_s: Int = -1
](
    zp: ImmPointer[Float32, _],
    ap: MutPointer[Float32, _],
    rows_rt: Int,
    cols: Int,
):
    """Vectorizes softmax across contiguous columns (SIMD_W columns at a time).
    """
    var rows = rows_s if rows_s > 0 else rows_rt
    var c = 0
    var c_vec_end = (cols // SIMD_W) * SIMD_W
    while c < c_vec_end:
        var col_max = zp.unsafe_load[width=SIMD_W](c)
        for r in range(1, rows):
            col_max = max(col_max, zp.unsafe_load[width=SIMD_W](r * cols + c))

        var sum_exp = SIMD[DType.float32, SIMD_W](0.0)
        for r in range(rows):
            var e_val = exp(
                zp.unsafe_load[width=SIMD_W](r * cols + c) - col_max
            )
            ap.unsafe_store[width=SIMD_W](r * cols + c, e_val)
            sum_exp += e_val

        var inv_sum = SIMD[DType.float32, SIMD_W](1.0) / sum_exp
        for r in range(rows):
            var ev = ap.unsafe_load[width=SIMD_W](r * cols + c)
            ap.unsafe_store[width=SIMD_W](r * cols + c, ev * inv_sum)
        c += SIMD_W

    while c < cols:
        var col_max = zp[unsafe_offset=c]
        for r in range(1, rows):
            col_max = max(col_max, zp[unsafe_offset=r * cols + c])
        var sum_exp = Float32(0.0)
        for r in range(rows):
            var e_val = exp(zp[unsafe_offset=r * cols + c] - col_max)
            ap[unsafe_offset=r * cols + c] = e_val
            sum_exp += e_val
        var inv_sum = Float32(1.0) / sum_exp
        for r in range(rows):
            ap[unsafe_offset=r * cols + c] *= inv_sum
        c += 1


def cpu_softmax_forward(z: CPUMatrix, mut a: CPUMatrix) raises:
    """Column-wise numerically stable softmax: A[:, c] = softmax(Z[:, c])."""
    if z.rows != a.rows or z.cols != a.cols:
        raise Error("cpu_softmax_forward: shape mismatch")
    var rows = z.rows
    var cols = z.cols
    var zp = z.data.unsafe_ptr()
    var ap = a.data.unsafe_ptr()

    if rows * cols < PARALLEL_ELEM_THRESHOLD:
        _softmax_forward_simd_cols(zp, ap, rows, cols)
        return

    @inline(.always)
    def col_worker(c: Int) {imm zp, imm ap, imm rows, imm cols}:
        var col_max = zp[unsafe_offset=c]
        for r in range(1, rows):
            var v = zp[unsafe_offset=r * cols + c]
            if v > col_max:
                col_max = v

        var sum_exp = Float32(0.0)
        for r in range(rows):
            var e_val = exp(zp[unsafe_offset=r * cols + c] - col_max)
            ap[unsafe_offset=r * cols + c] = e_val
            sum_exp += e_val

        var inv_sum = Float32(1.0) / sum_exp
        for r in range(rows):
            ap[unsafe_offset=r * cols + c] *= inv_sum

    parallelize(col_worker, cols)


def cpu_softmax_ce_grad(
    a2: CPUMatrix, labels: Span[Int, _], mut dz2: CPUMatrix
) raises -> Float32:
    """Computes dZ2 = A2 - one_hot(labels) and returns mean cross-entropy loss.
    """
    var rows = a2.rows
    var cols = a2.cols
    if dz2.rows != rows or dz2.cols != cols or len(labels) < cols:
        raise Error("cpu_softmax_ce_grad: shape mismatch")

    var ap = a2.data.unsafe_ptr()
    var dp = dz2.data.unsafe_ptr()
    var lp = labels.unsafe_ptr()

    if rows * cols < PARALLEL_ELEM_THRESHOLD:
        var total_loss = Float32(0.0)
        for c in range(cols):
            var lbl = lp[unsafe_offset=c]
            for r in range(rows):
                var idx = r * cols + c
                var p = ap[unsafe_offset=idx]
                if r == lbl:
                    dp[unsafe_offset=idx] = p - Float32(1.0)
                    total_loss += -log(max(p, Float32(1e-7)))
                else:
                    dp[unsafe_offset=idx] = p
        return total_loss / Float32(cols)

    var col_losses = List[Float32](length=cols, fill=Float32(0.0))
    var loss_ptr = col_losses.unsafe_ptr()

    @inline(.always)
    def col_worker(
        c: Int,
    ) {imm ap, imm dp, imm lp, imm loss_ptr, imm rows, imm cols}:
        var lbl = lp[unsafe_offset=c]
        for r in range(rows):
            var idx = r * cols + c
            var p = ap[unsafe_offset=idx]
            if r == lbl:
                dp[unsafe_offset=idx] = p - Float32(1.0)
                var clamped = max(p, Float32(1e-7))
                loss_ptr[unsafe_offset=c] = -log(clamped)
            else:
                dp[unsafe_offset=idx] = p

    parallelize(col_worker, cols)

    var total_loss = Float32(0.0)
    for c in range(cols):
        total_loss += col_losses[c]
    return total_loss / Float32(cols)


@inline(.always)
def cpu_softmax_ce_grad_and_metrics[
    rows: Int, cols: Int
](
    z2: StaticMatrixView[rows, cols, _],
    labels: Span[Int, _],
    dz2: StaticMatrixView[mut=True, rows, cols, _],
) -> Tuple[Float32, Int]:
    """Fuses softmax, cross-entropy loss, gradient dZ2, and correct count in registers.
    """
    comptime inv_cols = Float32(1.0) / Float32(cols)
    var zp = z2.ptr
    var dp = dz2.ptr
    var lp = labels.unsafe_ptr()

    var total_loss = Float32(0.0)
    var total_correct = 0

    var c = 0
    var c_vec_end = (cols // SIMD_W) * SIMD_W
    while c < c_vec_end:
        var col_max = zp.unsafe_load[width=SIMD_W](c)
        comptime for r in range(1, rows):
            col_max = max(col_max, zp.unsafe_load[width=SIMD_W](r * cols + c))

        var exps = Array[SIMD[DType.float32, SIMD_W], rows](uninitialized=True)
        var sum_exp = SIMD[DType.float32, SIMD_W](0.0)
        comptime for r in range(rows):
            var e_val = exp(
                zp.unsafe_load[width=SIMD_W](r * cols + c) - col_max
            )
            exps[r] = e_val
            sum_exp += e_val

        var inv_sum = SIMD[DType.float32, SIMD_W](1.0) / sum_exp
        var probs = Array[SIMD[DType.float32, SIMD_W], rows](uninitialized=True)
        comptime for r in range(rows):
            var p = exps[r] * inv_sum
            probs[r] = p
            dp.unsafe_store[width=SIMD_W](r * cols + c, p)

        # Argmax, loss, and one-hot gradient subtraction across the SIMD lanes
        comptime for lane in range(SIMD_W):
            var col_idx = c + lane
            var lbl = lp[unsafe_offset=col_idx]
            var best_r = 0
            var best_p = probs[0][lane]
            comptime for r in range(1, rows):
                var pv = probs[r][lane]
                if pv > best_p:
                    best_p = pv
                    best_r = r
            if best_r == lbl:
                total_correct += 1
            var p_lbl = probs[lbl][lane]
            dp[unsafe_offset=lbl * cols + col_idx] = p_lbl - Float32(1.0)
            total_loss += -log(max(p_lbl, Float32(1e-7)))

        c += SIMD_W

    while c < cols:
        var col_max = zp[unsafe_offset=c]
        var best_r = 0
        comptime for r in range(1, rows):
            var v = zp[unsafe_offset=r * cols + c]
            if v > col_max:
                col_max = v
                best_r = r

        var lbl = lp[unsafe_offset=c]
        if best_r == lbl:
            total_correct += 1

        var exps = Array[Float32, rows](uninitialized=True)
        var sum_exp = Float32(0.0)
        comptime for r in range(rows):
            var e_val = exp(zp[unsafe_offset=r * cols + c] - col_max)
            exps[r] = e_val
            sum_exp += e_val

        var inv_sum = Float32(1.0) / sum_exp
        comptime for r in range(rows):
            var p = exps[r] * inv_sum
            if r == lbl:
                dp[unsafe_offset=r * cols + c] = p - Float32(1.0)
                total_loss += -log(max(p, Float32(1e-7)))
            else:
                dp[unsafe_offset=r * cols + c] = p
        c += 1

    return total_loss * inv_cols, total_correct


@inline(.always)
def cpu_softmax_ce_grad_metrics_rows[
    batch_size: Int, num_classes: Int
](
    z2: StaticMatrixView[batch_size, num_classes, _],
    labels: Span[Int, _],
    dz2: StaticMatrixView[mut=True, batch_size, num_classes, _],
    db2: StaticMatrixView[mut=True, num_classes, 1, _],
) -> Tuple[Float32, Int]:
    """Fused row-major softmax, CE loss, accuracy count, dZ2, and dB2 accumulation.
    """
    comptime inv_batch = Float32(1.0) / Float32(batch_size)
    var zp = z2.ptr
    var dp = dz2.ptr
    var lp = labels.unsafe_ptr()
    var total_loss = Float32(0.0)
    var total_correct = 0

    var db2_acc = Array[Float32, num_classes](fill=Float32(0.0))

    for b in range(batch_size):
        var z_row = zp.unsafe_offset(b * num_classes)
        var d_row = dp.unsafe_offset(b * num_classes)
        var row_max = z_row[unsafe_offset=0]
        var best_r = 0
        comptime for r in range(1, num_classes):
            var v = z_row[unsafe_offset=r]
            if v > row_max:
                row_max = v
                best_r = r

        var lbl = lp[unsafe_offset=b]
        if best_r == lbl:
            total_correct += 1

        var exps = Array[Float32, num_classes](uninitialized=True)
        var sum_exp = Float32(0.0)
        comptime for r in range(num_classes):
            var e_val = exp(z_row[unsafe_offset=r] - row_max)
            exps[r] = e_val
            sum_exp += e_val

        var inv_sum = Float32(1.0) / sum_exp
        comptime for r in range(num_classes):
            var p = exps[r] * inv_sum
            var g = p - Float32(1.0) if r == lbl else p
            if r == lbl:
                total_loss += -log(max(p, Float32(1e-7)))
            d_row[unsafe_offset=r] = g
            db2_acc[r] += g

    var db2_p = db2.ptr
    comptime for r in range(num_classes):
        db2_p[unsafe_offset=r] = db2_acc[r] * inv_batch

    return total_loss * inv_batch, total_correct


@inline(.always)
def _relu_backward_mask_impl(
    zp: ImmPointer[Float32, _], dp: MutPointer[Float32, _], n: Int
):
    @inline(.always)
    def _mask[w: Int](i: Int) {imm zp, imm dp}:
        var zv = zp.unsafe_load[width=w](i)
        var dv = dp.unsafe_load[width=w](i)
        var mask = zv.gt(SIMD[DType.float32, w](0.0))
        dp.unsafe_store[width=w](
            i, mask.select(dv, SIMD[DType.float32, w](0.0))
        )

    vectorize[SIMD_W](n, _mask)


def cpu_relu_backward_mask(z: CPUMatrix, mut dz: CPUMatrix) raises:
    """Applies dZ = dZ * (Z > 0) in-place."""
    if z.rows != dz.rows or z.cols != dz.cols:
        raise Error("cpu_relu_backward_mask: shape mismatch")
    _relu_backward_mask_impl(
        z.data.unsafe_ptr(), dz.data.unsafe_ptr(), len(z.data)
    )


@inline(.always)
def cpu_relu_backward_mask_static[
    rows: Int, cols: Int
](
    a_or_z: StaticMatrixView[rows, cols, _],
    dz: StaticMatrixView[mut=True, rows, cols, _],
):
    _relu_backward_mask_impl(a_or_z.ptr, dz.ptr, rows * cols)


@inline(.always)
def cpu_scale_inplace(mut m: CPUMatrix, scale: Float32):
    var n = len(m.data)
    var mp = m.data.unsafe_ptr()

    @inline(.always)
    def _scale[w: Int](i: Int) {imm mp, imm scale}:
        var v = mp.unsafe_load[width=w](i)
        mp.unsafe_store[width=w](i, v * scale)

    vectorize[SIMD_W](n, _scale)


@inline(.always)
def _sum_axis1_scaled_impl(
    mp: ImmPointer[Float32, _],
    op: MutPointer[Float32, _],
    rows: Int,
    cols: Int,
    scale: Float32,
):
    @inline(.always)
    def row_worker(r: Int) {imm mp, imm op, imm cols, imm scale}:
        var row_ptr = mp.unsafe_offset(r * cols)
        var vec_acc = SIMD[DType.float32, SIMD_W](0.0)
        var c = 0
        var c_vec_end = (cols // SIMD_W) * SIMD_W
        while c < c_vec_end:
            vec_acc += row_ptr.unsafe_load[width=SIMD_W](c)
            c += SIMD_W
        var acc = vec_acc.reduce_add()
        while c < cols:
            acc += row_ptr[unsafe_offset=c]
            c += 1
        op[unsafe_offset=r] = acc * scale

    if rows * cols >= PARALLEL_ELEM_THRESHOLD:
        parallelize(row_worker, rows)
    else:
        for r in range(rows):
            row_worker(r)


def cpu_sum_axis1_scaled(
    m: CPUMatrix, scale: Float32, mut out: CPUMatrix
) raises:
    """Computes out[r, 0] = scale * sum_c m[r, c]."""
    if out.rows != m.rows or out.cols != 1:
        raise Error("cpu_sum_axis1_scaled: shape mismatch")
    _sum_axis1_scaled_impl(
        m.data.unsafe_ptr(), out.data.unsafe_ptr(), m.rows, m.cols, scale
    )


@inline(.always)
def cpu_sum_axis1_scaled_static[
    rows: Int, cols: Int
](
    m: StaticMatrixView[rows, cols, _],
    scale: Float32,
    out_v: StaticMatrixView[mut=True, rows, 1, _],
):
    _sum_axis1_scaled_impl(m.ptr, out_v.ptr, rows, cols, scale)


@inline(.always)
def cpu_sum_axis0_scaled_static[
    rows: Int, cols: Int
](
    m: StaticMatrixView[rows, cols, _],
    scale: Float32,
    out_v: StaticMatrixView[mut=True, cols, 1, _],
):
    """Computes out[c, 0] = scale * sum_r m[r, c] with contiguous SIMD streaming.
    """
    var mp = m.ptr
    var op = out_v.ptr
    var c = 0
    comptime c_vec4_end = (cols // SIMD_4W) * SIMD_4W
    comptime c_vec_end = (cols // SIMD_W) * SIMD_W

    comptime if SIMD_W <= 4:
        while c < c_vec4_end:
            var acc0 = SIMD[DType.float32, SIMD_W](0.0)
            var acc1 = SIMD[DType.float32, SIMD_W](0.0)
            var acc2 = SIMD[DType.float32, SIMD_W](0.0)
            var acc3 = SIMD[DType.float32, SIMD_W](0.0)
            for r in range(rows):
                var rp = mp.unsafe_offset(r * cols + c)
                acc0 += rp.unsafe_load[width=SIMD_W](0)
                acc1 += rp.unsafe_load[width=SIMD_W](SIMD_W)
                acc2 += rp.unsafe_load[width=SIMD_W](2 * SIMD_W)
                acc3 += rp.unsafe_load[width=SIMD_W](3 * SIMD_W)
            op.unsafe_store[width=SIMD_W](c, acc0 * scale)
            op.unsafe_store[width=SIMD_W](c + SIMD_W, acc1 * scale)
            op.unsafe_store[width=SIMD_W](c + 2 * SIMD_W, acc2 * scale)
            op.unsafe_store[width=SIMD_W](c + 3 * SIMD_W, acc3 * scale)
            c += SIMD_4W

    while c < c_vec_end:
        var acc = SIMD[DType.float32, SIMD_W](0.0)
        for r in range(rows):
            acc += mp.unsafe_load[width=SIMD_W](r * cols + c)
        op.unsafe_store[width=SIMD_W](c, acc * scale)
        c += SIMD_W

    comptime if (cols % SIMD_W) != 0:
        while c < cols:
            var s = Float32(0.0)
            for r in range(rows):
                s += mp[unsafe_offset=r * cols + c]
            op[unsafe_offset=c] = s * scale
            c += 1


@inline(.always)
def _sgd_momentum_update_impl[
    n_s: Int = -1
](
    pp: MutPointer[Float32, _],
    gp: ImmPointer[Float32, _],
    vp: MutPointer[Float32, _],
    n_rt: Int,
    lr: Float32,
    momentum: Float32,
):
    var n = n_s if n_s > 0 else n_rt
    var neg_lr = -lr

    @inline(.always)
    def update_range(
        start: Int, count: Int
    ) {imm pp, imm gp, imm vp, imm neg_lr, imm momentum}:
        var p_off = pp.unsafe_offset(start)
        var g_off = gp.unsafe_offset(start)
        var v_off = vp.unsafe_offset(start)

        @inline(.always)
        def _update[
            w: Int
        ](i: Int) {imm p_off, imm g_off, imm v_off, imm neg_lr, imm momentum}:
            var mom_v = SIMD[DType.float32, w](momentum)
            var nlr_v = SIMD[DType.float32, w](neg_lr)
            var v = mom_v.fma(
                v_off.unsafe_load[width=w](i), g_off.unsafe_load[width=w](i)
            )
            v_off.unsafe_store[width=w](i, v)
            p_off.unsafe_store[width=w](
                i, nlr_v.fma(v, p_off.unsafe_load[width=w](i))
            )

        vectorize[SIMD_W](count, _update)

    comptime if n_s >= 65_536:
        comptime CHUNK = 32_768
        comptime num_chunks = ceildiv(n_s, CHUNK)

        @inline(.always)
        def static_chunk_worker(c_idx: Int) {imm update_range}:
            var start = c_idx * CHUNK
            var count = min(CHUNK, n_s - start)
            update_range(start, count)

        parallelize(static_chunk_worker, num_chunks)
    elif n_s > 0:
        update_range(0, n_s)
    else:
        if n >= 65_536:
            comptime CHUNK = 16_384
            var num_chunks = ceildiv(n, CHUNK)

            @inline(.always)
            def dyn_chunk_worker(c_idx: Int) {imm update_range, imm n}:
                var start = c_idx * CHUNK
                var count = min(CHUNK, n - start)
                update_range(start, count)

            parallelize(dyn_chunk_worker, num_chunks)
        else:
            update_range(0, n)


def cpu_sgd_momentum_update(
    mut param: CPUMatrix,
    grad: CPUMatrix,
    mut velocity: CPUMatrix,
    lr: Float32,
    momentum: Float32,
) raises:
    """Updates velocity = momentum * velocity + grad; param -= lr * velocity."""
    if (
        param.rows != grad.rows
        or param.cols != grad.cols
        or param.rows != velocity.rows
        or param.cols != velocity.cols
    ):
        raise Error("cpu_sgd_momentum_update: shape mismatch")
    _sgd_momentum_update_impl(
        param.data.unsafe_ptr(),
        grad.data.unsafe_ptr(),
        velocity.data.unsafe_ptr(),
        len(param.data),
        lr,
        momentum,
    )


@inline(.always)
def cpu_sgd_momentum_update_static[
    rows: Int, cols: Int
](
    param: StaticMatrixView[mut=True, rows, cols, _],
    grad: StaticMatrixView[rows, cols, _],
    velocity: StaticMatrixView[mut=True, rows, cols, _],
    lr: Float32,
    momentum: Float32,
):
    _sgd_momentum_update_impl[rows * cols](
        param.ptr, grad.ptr, velocity.ptr, rows * cols, lr, momentum
    )


@inline(.always)
def cpu_argmax_axis0_static[
    rows: Int, cols: Int
](probs: StaticMatrixView[rows, cols, _], mut preds: List[Int]):
    var pp = probs.ptr
    var out_p = preds.unsafe_ptr()
    for c in range(cols):
        var best_r = 0
        var best_v = pp[unsafe_offset=c]
        comptime for r in range(1, rows):
            var v = pp[unsafe_offset=r * cols + c]
            if v > best_v:
                best_v = v
                best_r = r
        out_p[unsafe_offset=c] = best_r


def cpu_argmax_axis0(probs: CPUMatrix, mut preds: List[Int]) raises:
    """Computes argmax over rows (axis 0) for each column."""
    var rows = probs.rows
    var cols = probs.cols
    if len(preds) < cols:
        raise Error("cpu_argmax_axis0: preds buffer too small")
    var pp = probs.data.unsafe_ptr()
    var out_p = preds.unsafe_ptr()

    @inline(.always)
    def col_worker(c: Int) {imm pp, imm out_p, imm rows, imm cols}:
        var best_r = 0
        var best_v = pp[unsafe_offset=c]
        for r in range(1, rows):
            var v = pp[unsafe_offset=r * cols + c]
            if v > best_v:
                best_v = v
                best_r = r
        out_p[unsafe_offset=c] = best_r

    if rows * cols >= PARALLEL_ELEM_THRESHOLD:
        parallelize(col_worker, cols)
    else:
        for c in range(cols):
            col_worker(c)


@inline(.always)
def cpu_count_correct(
    preds: Span[Int, _], labels: Span[Int, _], count: Int
) -> Int:
    var pp = preds.unsafe_ptr()
    var lp = labels.unsafe_ptr()
    var total = 0

    @inline(.always)
    def _cnt[w: Int](i: Int) {mut total, imm pp, imm lp}:
        var pv = pp.unsafe_load[width=w](i)
        var lv = lp.unsafe_load[width=w](i)
        total += Int(pv.eq(lv).cast[DType.int64]().reduce_add())

    vectorize[SIMD_W](count, _cnt)
    return total


@inline(.always)
def cpu_eval_batch_logits_and_metrics[
    rows: Int, cols: Int
](
    z2: StaticMatrixView[rows, cols, _],
    mut out_preds: List[Int],
    offset: Int,
    labels: Span[Int, _],
    has_labels: Bool,
) -> Int:
    """Argmax directly on logits z2 (monotonic with softmax) without computing exp.
    """
    var zp = z2.ptr
    var out_p = out_preds.unsafe_ptr().unsafe_offset(offset)
    var correct = 0
    for c in range(cols):
        var best_r = 0
        var best_v = zp[unsafe_offset=c]
        comptime for r in range(1, rows):
            var v = zp[unsafe_offset=r * cols + c]
            if v > best_v:
                best_v = v
                best_r = r
        out_p[unsafe_offset=c] = best_r
        if has_labels and best_r == labels[offset + c]:
            correct += 1
    return correct


@inline(.always)
def cpu_eval_batch_logits_rows[
    batch_size: Int, num_classes: Int, has_labels: Bool
](
    z2: StaticMatrixView[batch_size, num_classes, _],
    mut out_preds: List[Int],
    offset: Int,
    labels: Span[Int, _],
) -> Int:
    """Row-major argmax directly on contiguous logits z2(batch_size, num_classes).
    """
    var zp = z2.ptr
    var out_p = out_preds.unsafe_ptr().unsafe_offset(offset)
    var correct = 0
    for b in range(batch_size):
        var z_row = zp.unsafe_offset(b * num_classes)
        var best_r = 0
        var best_v = z_row[unsafe_offset=0]
        comptime for r in range(1, num_classes):
            var v = z_row[unsafe_offset=r]
            if v > best_v:
                best_v = v
                best_r = r
        out_p[unsafe_offset=b] = best_r
        comptime if has_labels:
            if best_r == labels[offset + b]:
                correct += 1
    return correct


def cpu_predict_mlp_single(
    pixels: Span[Float32, _],
    w1: CPUMatrix,
    b1: CPUMatrix,
    w2: CPUMatrix,
    b2: CPUMatrix,
) raises -> Tuple[Int, List[Float32]]:
    """Zero-allocation-workspace single-sample MLP inference."""
    if (
        len(pixels) < w1.cols
        or w1.rows != b1.rows
        or w2.cols != w1.rows
        or w2.rows != b2.rows
    ):
        raise Error("cpu_predict_mlp_single: shape mismatch")

    var hidden = w1.rows
    var in_dim = w1.cols
    var out_dim = w2.rows

    var a1 = List[Float32](length=hidden, fill=Float32(0.0))
    _matmul_nn_matvec[EPILOGUE_BIAS_RELU](
        w1.data.unsafe_ptr(),
        pixels.unsafe_ptr(),
        a1.unsafe_ptr(),
        b1.data.unsafe_ptr(),
        hidden,
        in_dim,
    )

    var probs = List[Float32](length=out_dim, fill=Float32(0.0))
    _matmul_nn_matvec[EPILOGUE_BIAS](
        w2.data.unsafe_ptr(),
        a1.unsafe_ptr(),
        probs.unsafe_ptr(),
        b2.data.unsafe_ptr(),
        out_dim,
        hidden,
    )

    # In-place softmax + argmax over probs (contiguous out_dim elements)
    var pp = probs.unsafe_ptr()
    var max_v = pp[unsafe_offset=0]
    for r in range(1, out_dim):
        max_v = max(max_v, pp[unsafe_offset=r])

    var sum_exp = Float32(0.0)
    for r in range(out_dim):
        var e = exp(pp[unsafe_offset=r] - max_v)
        pp[unsafe_offset=r] = e
        sum_exp += e

    var inv_sum = Float32(1.0) / sum_exp
    var best_digit = 0
    var best_prob = Float32(-1.0)
    for r in range(out_dim):
        var p = pp[unsafe_offset=r] * inv_sum
        pp[unsafe_offset=r] = p
        if p > best_prob:
            best_prob = p
            best_digit = r

    return best_digit, probs^


def cpu_gather_batch_rows_shifted(
    pixels: Span[Float32, _],
    indices: Span[Int, _],
    batch_start: Int,
    batch_size: Int,
    mut out_x: CPUMatrix,
    mut out_labels: List[Int],
    all_labels: Span[Int, _],
    mut rng_state: UInt32,
):
    """Gathers shuffled 28x28 rows with random +/-1 pixel x/y shifts."""
    var px_p = pixels.unsafe_ptr()
    var idx_p = indices.unsafe_ptr()
    var lbl_p = all_labels.unsafe_ptr()
    var dst_p = out_x.data.unsafe_ptr()
    var out_lbl_p = out_labels.unsafe_ptr()
    var s = rng_state

    for b in range(batch_size):
        var src_idx = idx_p[unsafe_offset=batch_start + b]
        out_lbl_p[unsafe_offset=b] = lbl_p[unsafe_offset=src_idx]
        s ^= s << 13
        s ^= s >> 17
        s ^= s << 5
        var dy = Int(s % 3) - 1
        s ^= s << 13
        s ^= s >> 17
        s ^= s << 5
        var dx = Int(s % 3) - 1

        var src_row = px_p.unsafe_offset(src_idx * 784)
        var dst_row = dst_p.unsafe_offset(b * 784)

        if dy == 0 and dx == 0:
            unsafe_memcpy(dest=dst_row, src=src_row, count=784)
            continue

        for r in range(28):
            var sr = r - dy
            var dr_off = r * 28
            if sr < 0 or sr >= 28:
                for c in range(28):
                    dst_row[unsafe_offset=dr_off + c] = 0.0
            elif dx == 0:
                unsafe_memcpy(
                    dest=dst_row.unsafe_offset(dr_off),
                    src=src_row.unsafe_offset(sr * 28),
                    count=28,
                )
            elif dx == 1:
                dst_row[unsafe_offset=dr_off] = 0.0
                unsafe_memcpy(
                    dest=dst_row.unsafe_offset(dr_off + 1),
                    src=src_row.unsafe_offset(sr * 28),
                    count=27,
                )
            else:
                unsafe_memcpy(
                    dest=dst_row.unsafe_offset(dr_off),
                    src=src_row.unsafe_offset(sr * 28 + 1),
                    count=27,
                )
                dst_row[unsafe_offset=dr_off + 27] = 0.0

    rng_state = s


@inline(.always)
def cpu_apply_dropout_static[
    rows: Int, cols: Int
](
    a: StaticMatrixView[mut=True, rows, cols, _],
    seed_val: UInt32,
    keep_prob: Float32,
    inv_keep: Float32,
):
    """Applies inverted dropout in-place using a fast per-row PRNG."""
    var ap = a.ptr
    var thresh = UInt32(Float64(keep_prob) * 4294967295.0)

    @inline(.always)
    def row_worker(r: Int) {imm ap, imm seed_val, imm thresh, imm inv_keep}:
        var s = seed_val ^ UInt32(r * 747796405 + 2891336453)
        if s == 0:
            s = 1
        var row_p = ap.unsafe_offset(r * cols)
        for c in range(cols):
            s ^= s << 13
            s ^= s >> 17
            s ^= s << 5
            if s < thresh:
                row_p[unsafe_offset=c] *= inv_keep
            else:
                row_p[unsafe_offset=c] = 0.0

    comptime if rows * cols >= PARALLEL_ELEM_THRESHOLD:
        parallelize(row_worker, rows)
    else:
        for r in range(rows):
            row_worker(r)


@inline(.always)
def cpu_apply_relu_mask_scale_static[
    rows: Int, cols: Int
](
    dz: StaticMatrixView[mut=True, rows, cols, _],
    mask_act: StaticMatrixView[rows, cols, _],
    scale: Float32,
):
    """Computes dz = (mask_act > 0) ? dz * scale : 0.0 with SIMD."""
    comptime n = rows * cols
    var dp = dz.ptr
    var mp = mask_act.ptr

    @inline(.always)
    def _chunk(idx: Int, count: Int) {imm dp, imm mp, imm scale}:
        var d_sub = dp.unsafe_offset(idx)
        var m_sub = mp.unsafe_offset(idx)

        @inline(.always)
        def _vec[w: Int](i: Int) {imm d_sub, imm m_sub, imm scale}:
            var m = m_sub.unsafe_load[width=w](i)
            var v = d_sub.unsafe_load[width=w](i) * scale
            d_sub.unsafe_store[width=w](
                i, m.gt(0.0).select(v, SIMD[DType.float32, w](0.0))
            )

        vectorize[SIMD_W](count, _vec)

    comptime if n >= PARALLEL_ELEM_THRESHOLD:
        comptime chunk_sz = 32_768
        comptime num_tasks = ceildiv(n, chunk_sz)

        @inline(.always)
        def _worker(t: Int) {imm _chunk}:
            var start = t * chunk_sz
            _chunk(start, min(chunk_sz, n - start))

        parallelize(_worker, num_tasks)
    else:
        _chunk(0, n)


@inline(.always)
def cpu_softmax_ce_grad_metrics_smooth_rows[
    batch_size: Int, num_classes: Int
](
    z2: StaticMatrixView[batch_size, num_classes, _],
    labels: Span[Int, _],
    dz2: StaticMatrixView[mut=True, batch_size, num_classes, _],
    db2: StaticMatrixView[mut=True, num_classes, 1, _],
    smoothing: Float32,
) -> Tuple[Float32, Int]:
    """Sample-major softmax + label-smoothed CE loss, scaled dZ/dB, and accuracy.
    """
    comptime inv_batch = Float32(1.0) / Float32(batch_size)
    var off_val = smoothing / Float32(num_classes)
    var on_val = Float32(1.0) - smoothing + off_val

    var zp = z2.ptr
    var dzp = dz2.ptr
    var lp = labels.unsafe_ptr()
    var dbp = db2.ptr

    var total_loss = Float32(0.0)
    var correct = 0
    var acc_db = Array[Float32, num_classes](fill=Float32(0.0))

    for b in range(batch_size):
        var z_row = zp.unsafe_offset(b * num_classes)
        var dz_row = dzp.unsafe_offset(b * num_classes)
        var lbl = lp[unsafe_offset=b]

        var row_max = z_row[unsafe_offset=0]
        var best_c = 0
        comptime for c in range(1, num_classes):
            var v = z_row[unsafe_offset=c]
            if v > row_max:
                row_max = v
                best_c = c

        var sum_exp = Float32(0.0)
        var exps = Array[Float32, num_classes](uninitialized=True)
        comptime for c in range(num_classes):
            var e = exp(z_row[unsafe_offset=c] - row_max)
            exps[c] = e
            sum_exp += e

        var inv_sum = Float32(1.0) / sum_exp
        comptime for c in range(num_classes):
            var p = exps[c] * inv_sum
            var target = on_val if c == lbl else off_val
            total_loss -= target * log(max(p, Float32(1e-7)))
            var g = (p - target) * inv_batch
            dz_row[unsafe_offset=c] = g
            acc_db[c] += g

        if best_c == lbl:
            correct += 1

    comptime for c in range(num_classes):
        dbp[unsafe_offset=c] = acc_db[c]

    return total_loss * inv_batch, correct


@inline(.always)
def cpu_adamw_update_static[
    rows: Int, cols: Int
](
    param: StaticMatrixView[mut=True, rows, cols, _],
    grad: StaticMatrixView[rows, cols, _],
    m_state: StaticMatrixView[mut=True, rows, cols, _],
    v_state: StaticMatrixView[mut=True, rows, cols, _],
    lr: Float32,
    inv_bc1: Float32,
    inv_bc2: Float32,
    weight_decay: Float32,
):
    """SIMD-vectorized AdamW parameter update."""
    comptime n = rows * cols
    comptime beta1 = Float32(0.9)
    comptime beta2 = Float32(0.999)
    comptime one_m_b1 = Float32(1.0 - 0.9)
    comptime one_m_b2 = Float32(1.0 - 0.999)
    comptime eps = Float32(1e-8)

    var pp = param.ptr
    var gp = grad.ptr
    var mp = m_state.ptr
    var vp = v_state.ptr
    var decay = Float32(1.0) - lr * weight_decay

    @inline(.always)
    def _chunk(
        idx: Int, count: Int
    ) {
        imm pp,
        imm gp,
        imm mp,
        imm vp,
        imm lr,
        imm inv_bc1,
        imm inv_bc2,
        imm decay,
    }:
        var p_sub = pp.unsafe_offset(idx)
        var g_sub = gp.unsafe_offset(idx)
        var m_sub = mp.unsafe_offset(idx)
        var v_sub = vp.unsafe_offset(idx)

        @inline(.always)
        def _vec[
            w: Int
        ](i: Int) {
            imm p_sub,
            imm g_sub,
            imm m_sub,
            imm v_sub,
            imm lr,
            imm inv_bc1,
            imm inv_bc2,
            imm decay,
        }:
            var g = g_sub.unsafe_load[width=w](i)
            var m = m_sub.unsafe_load[width=w](i) * beta1 + g * one_m_b1
            var v = v_sub.unsafe_load[width=w](i) * beta2 + (g * g) * one_m_b2
            m_sub.unsafe_store[width=w](i, m)
            v_sub.unsafe_store[width=w](i, v)
            var m_hat = m * inv_bc1
            var v_hat = v * inv_bc2
            var p = p_sub.unsafe_load[width=w](i) * decay
            p = p - lr * m_hat / (sqrt(v_hat) + eps)
            p_sub.unsafe_store[width=w](i, p)

        vectorize[SIMD_W](count, _vec)

    comptime if n >= PARALLEL_ELEM_THRESHOLD:
        comptime chunk_sz = 32_768
        comptime num_tasks = ceildiv(n, chunk_sz)

        @inline(.always)
        def _worker(t: Int) {imm _chunk}:
            var start = t * chunk_sz
            _chunk(start, min(chunk_sz, n - start))

        parallelize(_worker, num_tasks)
    else:
        _chunk(0, n)


def cpu_predict_mlp3_single(
    pixels: Span[Float32, _],
    w1: CPUMatrix,
    b1: CPUMatrix,
    w2: CPUMatrix,
    b2: CPUMatrix,
    w3: CPUMatrix,
    b3: CPUMatrix,
) raises -> Tuple[Int, List[Float32]]:
    """Single-sample 3-layer MLP inference (w1: I x H1, w2: H1 x H2, w3: C x H2).
    """
    var in_dim = w1.rows
    var h1 = w1.cols
    var h2 = w2.cols
    var out_dim = w3.rows
    if len(pixels) < in_dim or w2.rows != h1 or w3.cols != h2:
        raise Error("cpu_predict_mlp3_single: shape mismatch")

    var a1 = List[Float32](length=h1, fill=Float32(0.0))
    _matmul_nn_impl[epilogue=EPILOGUE_BIAS_ROW_RELU](
        pixels.unsafe_ptr(),
        w1.data.unsafe_ptr(),
        a1.unsafe_ptr(),
        b1.data.unsafe_ptr(),
        1,
        h1,
        in_dim,
    )

    var a2 = List[Float32](length=h2, fill=Float32(0.0))
    _matmul_nn_impl[epilogue=EPILOGUE_BIAS_ROW_RELU](
        a1.unsafe_ptr(),
        w2.data.unsafe_ptr(),
        a2.unsafe_ptr(),
        b2.data.unsafe_ptr(),
        1,
        h2,
        h1,
    )

    var probs = List[Float32](length=out_dim, fill=Float32(0.0))
    _matmul_nn_matvec[EPILOGUE_BIAS](
        w3.data.unsafe_ptr(),
        a2.unsafe_ptr(),
        probs.unsafe_ptr(),
        b3.data.unsafe_ptr(),
        out_dim,
        h2,
    )

    var pp = probs.unsafe_ptr()
    var max_v = pp[unsafe_offset=0]
    for r in range(1, out_dim):
        max_v = max(max_v, pp[unsafe_offset=r])

    var sum_exp = Float32(0.0)
    for r in range(out_dim):
        var e = exp(pp[unsafe_offset=r] - max_v)
        pp[unsafe_offset=r] = e
        sum_exp += e

    var inv_sum = Float32(1.0) / sum_exp
    var best_digit = 0
    var best_prob = Float32(-1.0)
    for r in range(out_dim):
        var p = pp[unsafe_offset=r] * inv_sum
        pp[unsafe_offset=r] = p
        if p > best_prob:
            best_prob = p
            best_digit = r

    return best_digit, probs^
