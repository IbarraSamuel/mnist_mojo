from std.collections import Array
from std.math import align_up, ceildiv, divmod, exp, log, max, sqrt
from max.gpu import (
    WARP_SIZE,
    block_dim,
    block_idx,
    global_idx,
    lane_id,
    thread_idx,
)
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives import warp
from max.gpu.sync import barrier
from layout import (
    ImmTileTensor,
    MutTileTensor,
    TileTensor,
    row_major,
    stack_allocation,
)

from gpu_mem import DeviceMatrix


comptime TILE_SIZE = 16
comptime BIG_TILE = 32
comptime BLOCK_1D = 256
comptime HOST_WARP_ALIGN = 64
comptime GPU_EPILOGUE_NONE = 0
comptime GPU_EPILOGUE_BIAS = 1
comptime GPU_EPILOGUE_BIAS_RELU = 2
comptime GPU_EPILOGUE_BIAS_ROW = 3
comptime GPU_EPILOGUE_BIAS_ROW_RELU = 4
comptime GPU_EPILOGUE_RELU_MASK = 5


def matmul_nn_kernel[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
    epilogue: Int = GPU_EPILOGUE_NONE,
    A_ROWS: Int = M,
    BIAS_LEN: Int = M,
](
    a: ImmTileTensor[
        dtype, type_of(row_major[A_ROWS, K]()), ImmUntrackedOrigin
    ],
    b: ImmTileTensor[dtype, type_of(row_major[K, N]()), ImmUntrackedOrigin],
    bias: ImmTileTensor[
        dtype, type_of(row_major[BIAS_LEN, 1]()), ImmUntrackedOrigin
    ],
    mask_act: ImmTileTensor[
        dtype, type_of(row_major[M, N]()), ImmUntrackedOrigin
    ],
    c: MutTileTensor[dtype, type_of(row_major[M, N]()), MutUntrackedOrigin],
    a_row_offset: Int32 = 0,
):
    """Tiled matrix multiplication: C = A @ B (+ optional fused bias / ReLU / mask).
    """
    comptime assert (
        a.flat_rank == 2
        and b.flat_rank == 2
        and bias.flat_rank == 2
        and mask_act.flat_rank == 2
        and c.flat_rank == 2
    )
    comptime m_div = (M % TILE_SIZE == 0)
    comptime n_div = (N % TILE_SIZE == 0)
    comptime k_div = (K % TILE_SIZE == 0)
    comptime k_full_end = (K // TILE_SIZE) * TILE_SIZE
    comptime row_off_zero = (A_ROWS == M)

    var tx = thread_idx.x
    var ty = thread_idx.y
    var row = block_idx.y * TILE_SIZE + ty
    var col = block_idx.x * TILE_SIZE + tx
    var a_row = row if row_off_zero else Int(a_row_offset) + row

    # +1 column padding eliminates shared-memory bank conflicts on column reads.
    var sa = stack_allocation[dtype, address_space=.SHARED](
        row_major[TILE_SIZE, TILE_SIZE + 1]()
    )
    var sb = stack_allocation[dtype, address_space=.SHARED](
        row_major[TILE_SIZE, TILE_SIZE + 1]()
    )

    var acc: c.ElementType = 0.0

    for k_base in range(0, k_full_end, TILE_SIZE):
        comptime if m_div and row_off_zero:
            sa[ty, tx] = a[a_row, k_base + tx]
        else:
            if row < M and a_row < A_ROWS:
                sa[ty, tx] = a[a_row, k_base + tx]
            else:
                sa[ty, tx] = 0.0

        comptime if n_div:
            sb[ty, tx] = b[k_base + ty, col]
        else:
            if col < N:
                sb[ty, tx] = b[k_base + ty, col]
            else:
                sb[ty, tx] = 0.0

        barrier()

        comptime for k in range(TILE_SIZE):
            acc += sa[ty, k] * sb[k, tx]

        barrier()

    comptime if not k_div:
        comptime k_base = k_full_end
        if row < M and a_row < A_ROWS and k_base + tx < K:
            sa[ty, tx] = a[a_row, k_base + tx]
        else:
            sa[ty, tx] = 0.0

        if k_base + ty < K and col < N:
            sb[ty, tx] = b[k_base + ty, col]
        else:
            sb[ty, tx] = 0.0

        barrier()

        comptime for k in range(TILE_SIZE):
            acc += sa[ty, k] * sb[k, tx]

        barrier()

    @inline(.always)
    def write_out(
        val: c.ElementType,
    ) {imm row, imm col, imm bias, imm mask_act, imm c}:
        var out = val
        comptime if epilogue == GPU_EPILOGUE_BIAS:
            out = out + rebind[c.ElementType](bias[row, 0])
        elif epilogue == GPU_EPILOGUE_BIAS_RELU:
            out = max(
                out + rebind[c.ElementType](bias[row, 0]),
                c.ElementType(0.0),
            )
        elif epilogue == GPU_EPILOGUE_BIAS_ROW:
            out = out + rebind[c.ElementType](bias[col, 0])
        elif epilogue == GPU_EPILOGUE_BIAS_ROW_RELU:
            out = max(
                out + rebind[c.ElementType](bias[col, 0]),
                c.ElementType(0.0),
            )
        elif epilogue == GPU_EPILOGUE_RELU_MASK:
            if rebind[Scalar[dtype]](mask_act[row, col]) <= 0.0:
                out = c.ElementType(0.0)
        c[row, col] = out

    comptime if m_div and n_div:
        write_out(acc)
    else:
        if row < M and col < N:
            write_out(acc)


def matmul_nn_32x32_kernel[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
    epilogue: Int = GPU_EPILOGUE_NONE,
    A_ROWS: Int = M,
    BIAS_LEN: Int = M,
](
    a: ImmTileTensor[
        dtype, type_of(row_major[A_ROWS, K]()), ImmUntrackedOrigin
    ],
    b: ImmTileTensor[dtype, type_of(row_major[K, N]()), ImmUntrackedOrigin],
    bias: ImmTileTensor[
        dtype, type_of(row_major[BIAS_LEN, 1]()), ImmUntrackedOrigin
    ],
    mask_act: ImmTileTensor[
        dtype, type_of(row_major[M, N]()), ImmUntrackedOrigin
    ],
    c: MutTileTensor[dtype, type_of(row_major[M, N]()), MutUntrackedOrigin],
    a_row_offset: Int32 = 0,
):
    """32x32 output-tiled matmul (16x16 threads, 2x2 register micro-tile)."""
    comptime assert (
        a.flat_rank == 2
        and b.flat_rank == 2
        and bias.flat_rank == 2
        and mask_act.flat_rank == 2
        and c.flat_rank == 2
    )
    comptime m_div = (M % BIG_TILE == 0)
    comptime n_div = (N % BIG_TILE == 0)
    comptime k_div = (K % TILE_SIZE == 0)
    comptime k_full_end = (K // TILE_SIZE) * TILE_SIZE
    comptime row_off_zero = (A_ROWS == M)

    var tx = thread_idx.x
    var ty = thread_idx.y
    var row0 = block_idx.y * BIG_TILE + ty
    var row1 = row0 + TILE_SIZE
    var col0 = block_idx.x * BIG_TILE + tx
    var col1 = col0 + TILE_SIZE
    var a_row0 = row0 if row_off_zero else Int(a_row_offset) + row0
    var a_row1 = row1 if row_off_zero else Int(a_row_offset) + row1

    var sa = stack_allocation[dtype, address_space=.SHARED](
        row_major[BIG_TILE, TILE_SIZE + 1]()
    )
    var sb = stack_allocation[dtype, address_space=.SHARED](
        row_major[TILE_SIZE, BIG_TILE + 1]()
    )

    var acc00: c.ElementType = 0.0
    var acc01: c.ElementType = 0.0
    var acc10: c.ElementType = 0.0
    var acc11: c.ElementType = 0.0

    for k_base in range(0, k_full_end, TILE_SIZE):
        comptime if m_div and row_off_zero:
            sa[ty, tx] = a[a_row0, k_base + tx]
            sa[ty + TILE_SIZE, tx] = a[a_row1, k_base + tx]
        else:
            sa[ty, tx] = a[a_row0, k_base + tx] if (
                row0 < M and a_row0 < A_ROWS
            ) else c.ElementType(0.0)
            sa[ty + TILE_SIZE, tx] = a[a_row1, k_base + tx] if (
                row1 < M and a_row1 < A_ROWS
            ) else c.ElementType(0.0)

        comptime if n_div:
            sb[ty, tx] = b[k_base + ty, col0]
            sb[ty, tx + TILE_SIZE] = b[k_base + ty, col1]
        else:
            sb[ty, tx] = b[k_base + ty, col0] if col0 < N else c.ElementType(
                0.0
            )
            sb[ty, tx + TILE_SIZE] = b[
                k_base + ty, col1
            ] if col1 < N else c.ElementType(0.0)

        barrier()

        comptime for k in range(TILE_SIZE):
            var a0 = sa[ty, k]
            var a1 = sa[ty + TILE_SIZE, k]
            var b0 = sb[k, tx]
            var b1 = sb[k, tx + TILE_SIZE]
            acc00 += a0 * b0
            acc01 += a0 * b1
            acc10 += a1 * b0
            acc11 += a1 * b1

        barrier()

    comptime if not k_div:
        comptime k_base = k_full_end
        var k_valid_x = k_base + tx < K
        var k_valid_y = k_base + ty < K
        sa[ty, tx] = a[a_row0, k_base + tx] if (
            row0 < M and a_row0 < A_ROWS and k_valid_x
        ) else c.ElementType(0.0)
        sa[ty + TILE_SIZE, tx] = a[a_row1, k_base + tx] if (
            row1 < M and a_row1 < A_ROWS and k_valid_x
        ) else c.ElementType(0.0)
        sb[ty, tx] = b[k_base + ty, col0] if (
            k_valid_y and col0 < N
        ) else c.ElementType(0.0)
        sb[ty, tx + TILE_SIZE] = b[k_base + ty, col1] if (
            k_valid_y and col1 < N
        ) else c.ElementType(0.0)

        barrier()

        comptime for k in range(TILE_SIZE):
            var a0 = sa[ty, k]
            var a1 = sa[ty + TILE_SIZE, k]
            var b0 = sb[k, tx]
            var b1 = sb[k, tx + TILE_SIZE]
            acc00 += a0 * b0
            acc01 += a0 * b1
            acc10 += a1 * b0
            acc11 += a1 * b1

        barrier()

    @inline(.always)
    def write_cell(
        r: Int, c_idx: Int, val: c.ElementType
    ) {imm bias, imm mask_act, imm c}:
        var out = val
        comptime if epilogue == GPU_EPILOGUE_BIAS:
            out = out + rebind[c.ElementType](bias[r, 0])
        elif epilogue == GPU_EPILOGUE_BIAS_RELU:
            out = max(
                out + rebind[c.ElementType](bias[r, 0]),
                c.ElementType(0.0),
            )
        elif epilogue == GPU_EPILOGUE_BIAS_ROW:
            out = out + rebind[c.ElementType](bias[c_idx, 0])
        elif epilogue == GPU_EPILOGUE_BIAS_ROW_RELU:
            out = max(
                out + rebind[c.ElementType](bias[c_idx, 0]),
                c.ElementType(0.0),
            )
        elif epilogue == GPU_EPILOGUE_RELU_MASK:
            if rebind[Scalar[dtype]](mask_act[r, c_idx]) <= 0.0:
                out = c.ElementType(0.0)
        c[r, c_idx] = out

    comptime if m_div and n_div:
        write_cell(row0, col0, acc00)
        write_cell(row0, col1, acc01)
        write_cell(row1, col0, acc10)
        write_cell(row1, col1, acc11)
    else:
        if row0 < M:
            if col0 < N:
                write_cell(row0, col0, acc00)
            if col1 < N:
                write_cell(row0, col1, acc01)
        if row1 < M:
            if col0 < N:
                write_cell(row1, col0, acc10)
            if col1 < N:
                write_cell(row1, col1, acc11)


def matmul_nt_kernel[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
    epilogue: Int = GPU_EPILOGUE_NONE,
](
    a: ImmTileTensor[dtype, type_of(row_major[M, K]()), ImmUntrackedOrigin],
    b: ImmTileTensor[dtype, type_of(row_major[N, K]()), ImmUntrackedOrigin],
    bias: ImmTileTensor[dtype, type_of(row_major[N, 1]()), ImmUntrackedOrigin],
    c: MutTileTensor[dtype, type_of(row_major[M, N]()), MutUntrackedOrigin],
    scale: Scalar[dtype],
):
    """Tiled matrix multiplication: C = scale * (A @ B.T) (+ optional row bias).
    """
    comptime assert (
        a.flat_rank == 2
        and b.flat_rank == 2
        and bias.flat_rank == 2
        and c.flat_rank == 2
    )
    comptime m_div = (M % TILE_SIZE == 0)
    comptime n_div = (N % TILE_SIZE == 0)
    comptime k_div = (K % TILE_SIZE == 0)
    comptime k_full_end = (K // TILE_SIZE) * TILE_SIZE

    var tx = thread_idx.x
    var ty = thread_idx.y
    var row = block_idx.y * TILE_SIZE + ty
    var col = block_idx.x * TILE_SIZE + tx

    var sa = stack_allocation[dtype, address_space=.SHARED](
        row_major[TILE_SIZE, TILE_SIZE + 1]()
    )
    var sb = stack_allocation[dtype, address_space=.SHARED](
        row_major[TILE_SIZE, TILE_SIZE + 1]()
    )

    var acc: c.ElementType = 0.0

    for k_base in range(0, k_full_end, TILE_SIZE):
        comptime if m_div:
            sa[ty, tx] = a[row, k_base + tx]
        else:
            if row < M:
                sa[ty, tx] = a[row, k_base + tx]
            else:
                sa[ty, tx] = 0.0

        var b_row = block_idx.x * TILE_SIZE + ty
        var b_col = k_base + tx
        comptime if n_div:
            sb[ty, tx] = b[b_row, b_col]
        else:
            if b_row < N:
                sb[ty, tx] = b[b_row, b_col]
            else:
                sb[ty, tx] = 0.0

        barrier()

        comptime for k in range(TILE_SIZE):
            acc += sa[ty, k] * sb[tx, k]

        barrier()

    comptime if not k_div:
        comptime k_base = k_full_end
        if row < M and k_base + tx < K:
            sa[ty, tx] = a[row, k_base + tx]
        else:
            sa[ty, tx] = 0.0

        var b_row = block_idx.x * TILE_SIZE + ty
        var b_col = k_base + tx
        if b_row < N and b_col < K:
            sb[ty, tx] = b[b_row, b_col]
        else:
            sb[ty, tx] = 0.0

        barrier()

        comptime for k in range(TILE_SIZE):
            acc += sa[ty, k] * sb[tx, k]

        barrier()

    @inline(.always)
    def write_nt(
        val: c.ElementType,
    ) {imm row, imm col, imm bias, imm c, imm scale}:
        var out = val * scale
        comptime if epilogue == GPU_EPILOGUE_BIAS_ROW:
            out = out + rebind[c.ElementType](bias[col, 0])
        c[row, col] = out

    comptime if m_div and n_div:
        write_nt(acc)
    else:
        if row < M and col < N:
            write_nt(acc)


def matmul_tn_kernel[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
    has_relu_mask: Bool = False,
    has_scale: Bool = False,
](
    a: ImmTileTensor[dtype, type_of(row_major[K, M]()), ImmUntrackedOrigin],
    b: ImmTileTensor[dtype, type_of(row_major[K, N]()), ImmUntrackedOrigin],
    mask_act: ImmTileTensor[
        dtype, type_of(row_major[M, N]()), ImmUntrackedOrigin
    ],
    c: MutTileTensor[dtype, type_of(row_major[M, N]()), MutUntrackedOrigin],
    scale: Scalar[dtype] = 1.0,
):
    """Tiled matrix multiplication: C = scale * (A.T @ B) (with optional ReLU backward mask).
    """
    comptime assert (
        a.flat_rank == 2
        and b.flat_rank == 2
        and mask_act.flat_rank == 2
        and c.flat_rank == 2
    )
    comptime m_div = (M % TILE_SIZE == 0)
    comptime n_div = (N % TILE_SIZE == 0)
    comptime k_div = (K % TILE_SIZE == 0)
    comptime k_full_end = (K // TILE_SIZE) * TILE_SIZE

    var tx = thread_idx.x
    var ty = thread_idx.y
    var row = block_idx.y * TILE_SIZE + ty
    var col = block_idx.x * TILE_SIZE + tx

    var sa = stack_allocation[dtype, address_space=.SHARED](
        row_major[TILE_SIZE, TILE_SIZE + 1]()
    )
    var sb = stack_allocation[dtype, address_space=.SHARED](
        row_major[TILE_SIZE, TILE_SIZE + 1]()
    )

    var acc: c.ElementType = 0.0

    for k_base in range(0, k_full_end, TILE_SIZE):
        var a_row = k_base + ty
        var a_col = block_idx.y * TILE_SIZE + tx
        comptime if m_div:
            sa[ty, tx] = a[a_row, a_col]
        else:
            if a_col < M:
                sa[ty, tx] = a[a_row, a_col]
            else:
                sa[ty, tx] = 0.0

        comptime if n_div:
            sb[ty, tx] = b[k_base + ty, col]
        else:
            if col < N:
                sb[ty, tx] = b[k_base + ty, col]
            else:
                sb[ty, tx] = 0.0

        barrier()

        comptime for k in range(TILE_SIZE):
            acc += sa[k, ty] * sb[k, tx]

        barrier()

    comptime if not k_div:
        comptime k_base = k_full_end
        var a_row = k_base + ty
        var a_col = block_idx.y * TILE_SIZE + tx
        if a_row < K and a_col < M:
            sa[ty, tx] = a[a_row, a_col]
        else:
            sa[ty, tx] = 0.0

        if k_base + ty < K and col < N:
            sb[ty, tx] = b[k_base + ty, col]
        else:
            sb[ty, tx] = 0.0

        barrier()

        comptime for k in range(TILE_SIZE):
            acc += sa[k, ty] * sb[k, tx]

        barrier()

    @inline(.always)
    def write_tn(
        val: c.ElementType,
    ) {imm row, imm col, imm mask_act, imm c, imm scale}:
        comptime if has_relu_mask:
            if rebind[Scalar[dtype]](mask_act[row, col]) <= 0.0:
                c[row, col] = 0.0
                return
        comptime if has_scale:
            c[row, col] = val * scale
        else:
            c[row, col] = val

    comptime if m_div and n_div:
        write_tn(acc)
    else:
        if row < M and col < N:
            write_tn(acc)


def matmul_tn_32x32_kernel[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
    has_relu_mask: Bool = False,
    has_scale: Bool = False,
](
    a: ImmTileTensor[dtype, type_of(row_major[K, M]()), ImmUntrackedOrigin],
    b: ImmTileTensor[dtype, type_of(row_major[K, N]()), ImmUntrackedOrigin],
    mask_act: ImmTileTensor[
        dtype, type_of(row_major[M, N]()), ImmUntrackedOrigin
    ],
    c: MutTileTensor[dtype, type_of(row_major[M, N]()), MutUntrackedOrigin],
    scale: Scalar[dtype] = 1.0,
):
    """32x32 output-tiled TN matmul (16x16 threads, coalesced loads, 2x2 registers).
    """
    comptime assert (
        a.flat_rank == 2
        and b.flat_rank == 2
        and mask_act.flat_rank == 2
        and c.flat_rank == 2
    )
    comptime m_div = (M % BIG_TILE == 0)
    comptime n_div = (N % BIG_TILE == 0)
    comptime k_div = (K % TILE_SIZE == 0)
    comptime k_full_end = (K // TILE_SIZE) * TILE_SIZE

    var tx = thread_idx.x
    var ty = thread_idx.y
    var m_base = block_idx.y * BIG_TILE
    var n_base = block_idx.x * BIG_TILE
    var row0 = m_base + ty
    var row1 = row0 + TILE_SIZE
    var col0 = n_base + tx
    var col1 = col0 + TILE_SIZE

    var sa = stack_allocation[dtype, address_space=.SHARED](
        row_major[TILE_SIZE, BIG_TILE + 1]()
    )
    var sb = stack_allocation[dtype, address_space=.SHARED](
        row_major[TILE_SIZE, BIG_TILE + 1]()
    )

    var acc00: c.ElementType = 0.0
    var acc01: c.ElementType = 0.0
    var acc10: c.ElementType = 0.0
    var acc11: c.ElementType = 0.0

    for k_base in range(0, k_full_end, TILE_SIZE):
        var ak = k_base + ty
        comptime if m_div:
            sa[ty, tx] = a[ak, row0]
            sa[ty, tx + TILE_SIZE] = a[ak, row1]
        else:
            sa[ty, tx] = a[ak, m_base + tx] if (
                m_base + tx < M
            ) else c.ElementType(0.0)
            sa[ty, tx + TILE_SIZE] = a[ak, m_base + TILE_SIZE + tx] if (
                m_base + TILE_SIZE + tx < M
            ) else c.ElementType(0.0)

        comptime if n_div:
            sb[ty, tx] = b[ak, col0]
            sb[ty, tx + TILE_SIZE] = b[ak, col1]
        else:
            sb[ty, tx] = b[ak, col0] if col0 < N else c.ElementType(0.0)
            sb[ty, tx + TILE_SIZE] = b[ak, col1] if col1 < N else c.ElementType(
                0.0
            )

        barrier()

        comptime for k in range(TILE_SIZE):
            var a0 = sa[k, ty]
            var a1 = sa[k, ty + TILE_SIZE]
            var b0 = sb[k, tx]
            var b1 = sb[k, tx + TILE_SIZE]
            acc00 += a0 * b0
            acc01 += a0 * b1
            acc10 += a1 * b0
            acc11 += a1 * b1

        barrier()

    comptime if not k_div:
        comptime k_base = k_full_end
        var ak = k_base + ty
        var k_valid = ak < K
        var ac0 = m_base + tx
        var ac1 = ac0 + TILE_SIZE
        sa[ty, tx] = a[ak, ac0] if (k_valid and ac0 < M) else c.ElementType(0.0)
        sa[ty, tx + TILE_SIZE] = a[ak, ac1] if (
            k_valid and ac1 < M
        ) else c.ElementType(0.0)
        sb[ty, tx] = b[ak, col0] if (k_valid and col0 < N) else c.ElementType(
            0.0
        )
        sb[ty, tx + TILE_SIZE] = b[ak, col1] if (
            k_valid and col1 < N
        ) else c.ElementType(0.0)

        barrier()

        comptime for k in range(TILE_SIZE):
            var a0 = sa[k, ty]
            var a1 = sa[k, ty + TILE_SIZE]
            var b0 = sb[k, tx]
            var b1 = sb[k, tx + TILE_SIZE]
            acc00 += a0 * b0
            acc01 += a0 * b1
            acc10 += a1 * b0
            acc11 += a1 * b1

        barrier()

    @inline(.always)
    def write_tn32(
        r: Int, c_idx: Int, val: c.ElementType
    ) {imm mask_act, imm c, imm scale}:
        comptime if has_relu_mask:
            if rebind[Scalar[dtype]](mask_act[r, c_idx]) <= 0.0:
                c[r, c_idx] = 0.0
                return
        comptime if has_scale:
            c[r, c_idx] = val * scale
        else:
            c[r, c_idx] = val

    comptime if m_div and n_div:
        write_tn32(row0, col0, acc00)
        write_tn32(row0, col1, acc01)
        write_tn32(row1, col0, acc10)
        write_tn32(row1, col1, acc11)
    else:
        if row0 < M:
            if col0 < N:
                write_tn32(row0, col0, acc00)
            if col1 < N:
                write_tn32(row0, col1, acc01)
        if row1 < M:
            if col0 < N:
                write_tn32(row1, col0, acc10)
            if col1 < N:
                write_tn32(row1, col1, acc11)


def add_bias_relu_kernel[
    dtype: DType, rows: Int, cols: Int
](
    z: MutTileTensor[
        dtype, type_of(row_major[rows, cols]()), MutUntrackedOrigin
    ],
    bias: ImmTileTensor[
        dtype, type_of(row_major[rows, 1]()), ImmUntrackedOrigin
    ],
    a: MutTileTensor[
        dtype, type_of(row_major[rows, cols]()), MutUntrackedOrigin
    ],
):
    """Computes Z[r, c] += bias[r] and A[r, c] = max(Z[r, c], 0)."""
    comptime assert (
        z.flat_rank == 2 and bias.flat_rank == 2 and a.flat_rank == 2
    )
    var row = global_idx.y
    var col = global_idx.x
    if row < rows and col < cols:
        var val = z[row, col] + rebind[z.ElementType](bias[row, 0])
        z[row, col] = val
        a[row, col] = max(val, z.ElementType(0.0))


def add_bias_kernel[
    dtype: DType, rows: Int, cols: Int
](
    z: MutTileTensor[
        dtype, type_of(row_major[rows, cols]()), MutUntrackedOrigin
    ],
    bias: ImmTileTensor[
        dtype, type_of(row_major[rows, 1]()), ImmUntrackedOrigin
    ],
):
    """Computes Z[r, c] += bias[r] in-place."""
    comptime assert z.flat_rank == 2 and bias.flat_rank == 2
    var row = global_idx.y
    var col = global_idx.x
    if row < rows and col < cols:
        z[row, col] = z[row, col] + rebind[z.ElementType](bias[row, 0])


def softmax_col_kernel[
    dtype: DType, rows: Int, cols: Int
](
    z: ImmTileTensor[
        dtype, type_of(row_major[rows, cols]()), ImmUntrackedOrigin
    ],
    a: MutTileTensor[
        dtype, type_of(row_major[rows, cols]()), MutUntrackedOrigin
    ],
):
    """Per-column numerically stable softmax using register caching when small.
    """
    comptime assert (
        dtype.is_floating_point()
    ), "softmax requires floating-point dtype"
    comptime assert z.flat_rank == 2 and a.flat_rank == 2
    var col = global_idx.x
    if col >= cols:
        return

    comptime if rows <= 32:
        var vals = Array[Scalar[dtype], rows](uninitialized=True)
        vals[0] = rebind[Scalar[dtype]](z[0, col])
        var col_max = vals[0]
        comptime for r in range(1, rows):
            var v = rebind[Scalar[dtype]](z[r, col])
            vals[r] = v
            col_max = max(col_max, v)

        var sum_exp: Scalar[dtype] = 0.0
        comptime for r in range(rows):
            var e_val = exp(vals[r] - col_max)
            vals[r] = e_val
            sum_exp += e_val

        var inv_sum = Scalar[dtype](1.0) / sum_exp
        comptime for r in range(rows):
            a[r, col] = rebind[a.ElementType](vals[r] * inv_sum)
    else:
        var col_max = rebind[Scalar[dtype]](z[0, col])
        for r in range(1, rows):
            col_max = max(col_max, rebind[Scalar[dtype]](z[r, col]))

        var sum_exp: Scalar[dtype] = 0.0
        for r in range(rows):
            var e_val = exp(rebind[Scalar[dtype]](z[r, col]) - col_max)
            a[r, col] = rebind[a.ElementType](e_val)
            sum_exp += e_val

        var inv_sum = Scalar[dtype](1.0) / sum_exp
        for r in range(rows):
            a[r, col] = a[r, col] * inv_sum


def softmax_ce_grad_kernel[
    dtype: DType, rows: Int, cols: Int
](
    a2: ImmTileTensor[
        dtype, type_of(row_major[rows, cols]()), ImmUntrackedOrigin
    ],
    labels: ImmTileTensor[
        DType.int32, type_of(row_major[cols, 1]()), ImmUntrackedOrigin
    ],
    dz2: MutTileTensor[
        dtype, type_of(row_major[rows, cols]()), MutUntrackedOrigin
    ],
    losses: MutTileTensor[
        dtype, type_of(row_major[cols, 1]()), MutUntrackedOrigin
    ],
):
    """Computes dZ2 = A2 - one_hot(labels) and per-sample cross-entropy loss."""
    comptime assert (
        dtype.is_floating_point()
    ), "cross-entropy requires floating-point dtype"
    comptime assert (
        a2.flat_rank == 2
        and labels.flat_rank == 2
        and dz2.flat_rank == 2
        and losses.flat_rank == 2
    )
    var col = global_idx.x
    if col >= cols:
        return

    var lbl = Int(labels[col, 0])
    for r in range(rows):
        var p = rebind[Scalar[dtype]](a2[r, col])
        if r == lbl:
            dz2[r, col] = rebind[dz2.ElementType](p - Scalar[dtype](1.0))
            var clamped = max(p, Scalar[dtype](1e-7))
            losses[col, 0] = rebind[losses.ElementType](-log(clamped))
        else:
            dz2[r, col] = rebind[dz2.ElementType](p)


def relu_backward_mask_kernel[
    dtype: DType, rows: Int, cols: Int
](
    z: ImmTileTensor[
        dtype, type_of(row_major[rows, cols]()), ImmUntrackedOrigin
    ],
    dz: MutTileTensor[
        dtype, type_of(row_major[rows, cols]()), MutUntrackedOrigin
    ],
):
    """Computes dZ = dZ * (Z > 0) in-place using 1D indexing."""
    comptime assert z.flat_rank == 2 and dz.flat_rank == 2
    comptime total = rows * cols
    var tid = global_idx.x
    if tid < total:
        var row: Int
        var col: Int
        comptime if cols == 1:
            row = tid
            col = 0
        else:
            row, col = divmod(tid, cols)
        if rebind[Scalar[dtype]](z[row, col]) <= 0.0:
            dz[row, col] = 0.0


def scale_kernel[
    dtype: DType, rows: Int, cols: Int
](
    m: MutTileTensor[
        dtype, type_of(row_major[rows, cols]()), MutUntrackedOrigin
    ],
    scale: Scalar[dtype],
):
    comptime assert m.flat_rank == 2
    comptime total = rows * cols
    var tid = global_idx.x
    if tid < total:
        var row: Int
        var col: Int
        comptime if cols == 1:
            row = tid
            col = 0
        else:
            row, col = divmod(tid, cols)
        m[row, col] = m[row, col] * scale


def sum_axis1_scaled_kernel[
    dtype: DType, rows: Int, cols: Int, block_threads: Int
](
    m: ImmTileTensor[
        dtype, type_of(row_major[rows, cols]()), ImmUntrackedOrigin
    ],
    out_v: MutTileTensor[
        dtype, type_of(row_major[rows, 1]()), MutUntrackedOrigin
    ],
    scale: Scalar[dtype],
):
    """Reduces axis 1 (cols) for each row using warp reductions."""
    comptime assert m.flat_rank == 2 and out_v.flat_rank == 2
    var row = block_idx.x
    var tid = thread_idx.x
    if row >= rows:
        return

    var local_sum: Scalar[dtype] = 0.0
    var c = tid
    while c < cols:
        local_sum += rebind[Scalar[dtype]](m[row, c])
        c += block_threads

    var warp_sum = warp.sum(local_sum)
    comptime num_warps = block_threads // WARP_SIZE
    comptime if num_warps <= 1:
        if tid == 0:
            out_v[row, 0] = rebind[out_v.ElementType](warp_sum * scale)
    else:
        var shared = stack_allocation[dtype, address_space=.SHARED](
            row_major[num_warps]()
        )
        var wid = tid // WARP_SIZE
        if lane_id() == 0:
            shared[wid] = rebind[shared.ElementType](warp_sum)
        barrier()
        if tid == 0:
            var total: Scalar[dtype] = 0.0
            comptime for w in range(num_warps):
                total += rebind[Scalar[dtype]](shared[w])
            out_v[row, 0] = rebind[out_v.ElementType](total * scale)


def sgd_momentum_kernel[
    dtype: DType, rows: Int, cols: Int
](
    param: MutTileTensor[
        dtype, type_of(row_major[rows, cols]()), MutUntrackedOrigin
    ],
    grad: ImmTileTensor[
        dtype, type_of(row_major[rows, cols]()), ImmUntrackedOrigin
    ],
    vel: MutTileTensor[
        dtype, type_of(row_major[rows, cols]()), MutUntrackedOrigin
    ],
    lr: Scalar[dtype],
    momentum: Scalar[dtype],
):
    comptime assert (
        param.flat_rank == 2 and grad.flat_rank == 2 and vel.flat_rank == 2
    )
    comptime total = rows * cols
    var tid = global_idx.x
    if tid < total:
        var row: Int
        var col: Int
        comptime if cols == 1:
            row = tid
            col = 0
        else:
            row, col = divmod(tid, cols)
        var v = momentum * rebind[Scalar[dtype]](vel[row, col]) + rebind[
            Scalar[dtype]
        ](grad[row, col])
        vel[row, col] = rebind[vel.ElementType](v)
        param[row, col] = rebind[param.ElementType](
            rebind[Scalar[dtype]](param[row, col]) - lr * v
        )


def sgd_momentum_mlp_kernel[
    dtype: DType, W1R: Int, W1C: Int, B1N: Int, W2R: Int, W2C: Int, B2N: Int
](
    w1: MutTileTensor[
        dtype, type_of(row_major[W1R, W1C]()), MutUntrackedOrigin
    ],
    dw1: ImmTileTensor[
        dtype, type_of(row_major[W1R, W1C]()), ImmUntrackedOrigin
    ],
    vw1: MutTileTensor[
        dtype, type_of(row_major[W1R, W1C]()), MutUntrackedOrigin
    ],
    b1: MutTileTensor[dtype, type_of(row_major[B1N, 1]()), MutUntrackedOrigin],
    db1: ImmTileTensor[dtype, type_of(row_major[B1N, 1]()), ImmUntrackedOrigin],
    vb1: MutTileTensor[dtype, type_of(row_major[B1N, 1]()), MutUntrackedOrigin],
    w2: MutTileTensor[
        dtype, type_of(row_major[W2R, W2C]()), MutUntrackedOrigin
    ],
    dw2: ImmTileTensor[
        dtype, type_of(row_major[W2R, W2C]()), ImmUntrackedOrigin
    ],
    vw2: MutTileTensor[
        dtype, type_of(row_major[W2R, W2C]()), MutUntrackedOrigin
    ],
    b2: MutTileTensor[dtype, type_of(row_major[B2N, 1]()), MutUntrackedOrigin],
    db2: ImmTileTensor[dtype, type_of(row_major[B2N, 1]()), ImmUntrackedOrigin],
    vb2: MutTileTensor[dtype, type_of(row_major[B2N, 1]()), MutUntrackedOrigin],
    lr: Scalar[dtype],
    momentum: Scalar[dtype],
):
    """Fused SGD + momentum update for all 4 MLP parameter tensors in one launch.
    """
    comptime assert (
        w1.flat_rank == 2
        and dw1.flat_rank == 2
        and vw1.flat_rank == 2
        and b1.flat_rank == 2
        and db1.flat_rank == 2
        and vb1.flat_rank == 2
        and w2.flat_rank == 2
        and dw2.flat_rank == 2
        and vw2.flat_rank == 2
        and b2.flat_rank == 2
        and db2.flat_rank == 2
        and vb2.flat_rank == 2
    )
    comptime n_w1 = W1R * W1C
    comptime n_b1 = B1N
    comptime n_w2 = W2R * W2C
    comptime n_b2 = B2N
    comptime off_b1 = n_w1
    comptime off_w2 = off_b1 + n_b1
    comptime off_b2 = off_w2 + n_w2
    comptime total = off_b2 + n_b2

    var tid = global_idx.x
    if tid >= total:
        return

    if tid < off_b1:
        var r, c = divmod(tid, W1C)
        var v = momentum * rebind[Scalar[dtype]](vw1[r, c]) + rebind[
            Scalar[dtype]
        ](dw1[r, c])
        vw1[r, c] = rebind[vw1.ElementType](v)
        w1[r, c] = rebind[w1.ElementType](
            rebind[Scalar[dtype]](w1[r, c]) - lr * v
        )
    elif tid < off_w2:
        var r = tid - off_b1
        var v = momentum * rebind[Scalar[dtype]](vb1[r, 0]) + rebind[
            Scalar[dtype]
        ](db1[r, 0])
        vb1[r, 0] = rebind[vb1.ElementType](v)
        b1[r, 0] = rebind[b1.ElementType](
            rebind[Scalar[dtype]](b1[r, 0]) - lr * v
        )
    elif tid < off_b2:
        var idx = tid - off_w2
        var r, c = divmod(idx, W2C)
        var v = momentum * rebind[Scalar[dtype]](vw2[r, c]) + rebind[
            Scalar[dtype]
        ](dw2[r, c])
        vw2[r, c] = rebind[vw2.ElementType](v)
        w2[r, c] = rebind[w2.ElementType](
            rebind[Scalar[dtype]](w2[r, c]) - lr * v
        )
    else:
        var r = tid - off_b2
        var v = momentum * rebind[Scalar[dtype]](vb2[r, 0]) + rebind[
            Scalar[dtype]
        ](db2[r, 0])
        vb2[r, 0] = rebind[vb2.ElementType](v)
        b2[r, 0] = rebind[b2.ElementType](
            rebind[Scalar[dtype]](b2[r, 0]) - lr * v
        )


@inline(.always)
def _launch_sgd_momentum_mlp[
    dtype: DType, W1R: Int, W1C: Int, B1N: Int, W2R: Int, W2C: Int, B2N: Int
](
    ctx: DeviceContext,
    mut w1: DeviceMatrix[dtype, W1R, W1C],
    dw1: DeviceMatrix[dtype, W1R, W1C],
    mut vw1: DeviceMatrix[dtype, W1R, W1C],
    mut b1: DeviceMatrix[dtype, B1N, 1],
    db1: DeviceMatrix[dtype, B1N, 1],
    mut vb1: DeviceMatrix[dtype, B1N, 1],
    mut w2: DeviceMatrix[dtype, W2R, W2C],
    dw2: DeviceMatrix[dtype, W2R, W2C],
    mut vw2: DeviceMatrix[dtype, W2R, W2C],
    mut b2: DeviceMatrix[dtype, B2N, 1],
    db2: DeviceMatrix[dtype, B2N, 1],
    mut vb2: DeviceMatrix[dtype, B2N, 1],
    lr: Scalar[dtype],
    momentum: Scalar[dtype],
) raises:
    comptime total = W1R * W1C + B1N + W2R * W2C + B2N
    comptime kernel = sgd_momentum_mlp_kernel[
        dtype, W1R, W1C, B1N, W2R, W2C, B2N
    ]
    ctx.enqueue_function[kernel](
        TileTensor(w1.buffer, w1.layout),
        TileTensor(dw1.buffer, dw1.layout).as_imm(),
        TileTensor(vw1.buffer, vw1.layout),
        TileTensor(b1.buffer, b1.layout),
        TileTensor(db1.buffer, db1.layout).as_imm(),
        TileTensor(vb1.buffer, vb1.layout),
        TileTensor(w2.buffer, w2.layout),
        TileTensor(dw2.buffer, dw2.layout).as_imm(),
        TileTensor(vw2.buffer, vw2.layout),
        TileTensor(b2.buffer, b2.layout),
        TileTensor(db2.buffer, db2.layout).as_imm(),
        TileTensor(vb2.buffer, vb2.layout),
        lr,
        momentum,
        grid_dim=ceildiv(total, BLOCK_1D),
        block_dim=BLOCK_1D,
    )


@inline(.always)
def gpu_sgd_momentum_update_mlp[
    dtype: DType, H: Int, I: Int, C: Int
](
    ctx: DeviceContext,
    mut w1: DeviceMatrix[dtype, H, I],
    dw1: DeviceMatrix[dtype, H, I],
    mut vw1: DeviceMatrix[dtype, H, I],
    mut b1: DeviceMatrix[dtype, H, 1],
    db1: DeviceMatrix[dtype, H, 1],
    mut vb1: DeviceMatrix[dtype, H, 1],
    mut w2: DeviceMatrix[dtype, C, H],
    dw2: DeviceMatrix[dtype, C, H],
    mut vw2: DeviceMatrix[dtype, C, H],
    mut b2: DeviceMatrix[dtype, C, 1],
    db2: DeviceMatrix[dtype, C, 1],
    mut vb2: DeviceMatrix[dtype, C, 1],
    lr: Scalar[dtype],
    momentum: Scalar[dtype],
) raises:
    _launch_sgd_momentum_mlp(
        ctx,
        w1,
        dw1,
        vw1,
        b1,
        db1,
        vb1,
        w2,
        dw2,
        vw2,
        b2,
        db2,
        vb2,
        lr,
        momentum,
    )


@inline(.always)
def gpu_sgd_momentum_update_mlp_sm[
    dtype: DType, I: Int, H: Int, C: Int
](
    ctx: DeviceContext,
    mut w1: DeviceMatrix[dtype, I, H],
    dw1: DeviceMatrix[dtype, I, H],
    mut vw1: DeviceMatrix[dtype, I, H],
    mut b1: DeviceMatrix[dtype, H, 1],
    db1: DeviceMatrix[dtype, H, 1],
    mut vb1: DeviceMatrix[dtype, H, 1],
    mut w2: DeviceMatrix[dtype, C, H],
    dw2: DeviceMatrix[dtype, C, H],
    mut vw2: DeviceMatrix[dtype, C, H],
    mut b2: DeviceMatrix[dtype, C, 1],
    db2: DeviceMatrix[dtype, C, 1],
    mut vb2: DeviceMatrix[dtype, C, 1],
    lr: Scalar[dtype],
    momentum: Scalar[dtype],
) raises:
    _launch_sgd_momentum_mlp(
        ctx,
        w1,
        dw1,
        vw1,
        b1,
        db1,
        vb1,
        w2,
        dw2,
        vw2,
        b2,
        db2,
        vb2,
        lr,
        momentum,
    )


def argmax_axis0_kernel[
    dtype: DType, rows: Int, cols: Int
](
    probs: ImmTileTensor[
        dtype, type_of(row_major[rows, cols]()), ImmUntrackedOrigin
    ],
    preds: MutTileTensor[
        DType.int32, type_of(row_major[cols, 1]()), MutUntrackedOrigin
    ],
):
    """Per-column argmax over axis 0 with zero write races."""
    comptime assert probs.flat_rank == 2 and preds.flat_rank == 2
    var col = global_idx.x
    if col >= cols:
        return

    var best_r = Int32(0)
    var best_v = rebind[Scalar[dtype]](probs[0, col])
    for r in range(1, rows):
        var v = rebind[Scalar[dtype]](probs[r, col])
        if v > best_v:
            best_v = v
            best_r = Int32(r)
    preds[col, 0] = best_r


@inline(.always)
def _launch_matmul_nn[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
    epilogue: Int,
    A_ROWS: Int,
    BIAS_LEN: Int,
](
    ctx: DeviceContext,
    a: DeviceMatrix[dtype, A_ROWS, K],
    b: DeviceMatrix[dtype, K, N],
    bias_buf: DeviceBuffer[dtype],
    mask_buf: DeviceBuffer[dtype],
    mut c: DeviceMatrix[dtype, M, N],
    a_row_offset: Int32 = 0,
) raises:
    comptime use_32 = (M >= 64 and N >= 64)
    var ta = TileTensor(a.buffer, a.layout)
    var tb = TileTensor(b.buffer, b.layout)
    var tbias = TileTensor(bias_buf, row_major[BIAS_LEN, 1]())
    var tmask = TileTensor(mask_buf, row_major[M, N]())
    var tc = TileTensor(c.buffer, c.layout)
    comptime if use_32:
        comptime kernel = matmul_nn_32x32_kernel[
            dtype, M, N, K, epilogue, A_ROWS, BIAS_LEN
        ]
        ctx.enqueue_function[kernel](
            ta.as_imm(),
            tb.as_imm(),
            tbias.as_imm(),
            tmask.as_imm(),
            tc,
            a_row_offset,
            grid_dim=(ceildiv(N, BIG_TILE), ceildiv(M, BIG_TILE)),
            block_dim=(TILE_SIZE, TILE_SIZE),
        )
    else:
        comptime kernel = matmul_nn_kernel[
            dtype, M, N, K, epilogue, A_ROWS, BIAS_LEN
        ]
        ctx.enqueue_function[kernel](
            ta.as_imm(),
            tb.as_imm(),
            tbias.as_imm(),
            tmask.as_imm(),
            tc,
            a_row_offset,
            grid_dim=(ceildiv(N, TILE_SIZE), ceildiv(M, TILE_SIZE)),
            block_dim=(TILE_SIZE, TILE_SIZE),
        )


@inline(.always)
def gpu_matmul_nn[
    dtype: DType, M: Int, N: Int, K: Int
](
    ctx: DeviceContext,
    a: DeviceMatrix[dtype, M, K],
    b: DeviceMatrix[dtype, K, N],
    mut c: DeviceMatrix[dtype, M, N],
) raises:
    _launch_matmul_nn[dtype, M, N, K, GPU_EPILOGUE_NONE, M, M](
        ctx, a, b, a.buffer, a.buffer, c
    )


@inline(.always)
def gpu_matmul_nn_bias[
    dtype: DType, M: Int, N: Int, K: Int
](
    ctx: DeviceContext,
    a: DeviceMatrix[dtype, M, K],
    b: DeviceMatrix[dtype, K, N],
    bias: DeviceMatrix[dtype, M, 1],
    mut c: DeviceMatrix[dtype, M, N],
) raises:
    _launch_matmul_nn[dtype, M, N, K, GPU_EPILOGUE_BIAS, M, M](
        ctx, a, b, bias.buffer, a.buffer, c
    )


@inline(.always)
def gpu_matmul_nn_bias_relu[
    dtype: DType, M: Int, N: Int, K: Int
](
    ctx: DeviceContext,
    a: DeviceMatrix[dtype, M, K],
    b: DeviceMatrix[dtype, K, N],
    bias: DeviceMatrix[dtype, M, 1],
    mut c: DeviceMatrix[dtype, M, N],
) raises:
    _launch_matmul_nn[dtype, M, N, K, GPU_EPILOGUE_BIAS_RELU, M, M](
        ctx, a, b, bias.buffer, a.buffer, c
    )


@inline(.always)
def gpu_matmul_nn_bias_row_relu[
    dtype: DType, M: Int, N: Int, K: Int
](
    ctx: DeviceContext,
    a: DeviceMatrix[dtype, M, K],
    b: DeviceMatrix[dtype, K, N],
    bias: DeviceMatrix[dtype, N, 1],
    mut c: DeviceMatrix[dtype, M, N],
) raises:
    _launch_matmul_nn[dtype, M, N, K, GPU_EPILOGUE_BIAS_ROW_RELU, M, N](
        ctx, a, b, bias.buffer, a.buffer, c
    )


@inline(.always)
def gpu_matmul_nn_offset_bias_row_relu[
    dtype: DType, M: Int, N: Int, K: Int, A_ROWS: Int
](
    ctx: DeviceContext,
    a: DeviceMatrix[dtype, A_ROWS, K],
    a_row_offset: Int,
    b: DeviceMatrix[dtype, K, N],
    bias: DeviceMatrix[dtype, N, 1],
    mut c: DeviceMatrix[dtype, M, N],
) raises:
    _launch_matmul_nn[dtype, M, N, K, GPU_EPILOGUE_BIAS_ROW_RELU, A_ROWS, N](
        ctx, a, b, bias.buffer, a.buffer, c, Int32(a_row_offset)
    )


@inline(.always)
def gpu_matmul_nn_relu_mask[
    dtype: DType, M: Int, N: Int, K: Int
](
    ctx: DeviceContext,
    a: DeviceMatrix[dtype, M, K],
    b: DeviceMatrix[dtype, K, N],
    mask_act: DeviceMatrix[dtype, M, N],
    mut c: DeviceMatrix[dtype, M, N],
) raises:
    _launch_matmul_nn[dtype, M, N, K, GPU_EPILOGUE_RELU_MASK, M, M](
        ctx, a, b, a.buffer, mask_act.buffer, c
    )


@inline(.always)
def gpu_matmul_nt[
    dtype: DType, M: Int, N: Int, K: Int
](
    ctx: DeviceContext,
    a: DeviceMatrix[dtype, M, K],
    b: DeviceMatrix[dtype, N, K],
    mut c: DeviceMatrix[dtype, M, N],
    scale: Scalar[dtype] = 1.0,
) raises:
    comptime kernel = matmul_nt_kernel[dtype, M, N, K, GPU_EPILOGUE_NONE]
    var ta = TileTensor(a.buffer, a.layout)
    var tb = TileTensor(b.buffer, b.layout)
    var tdummy = TileTensor(a.buffer, row_major[N, 1]())
    var tc = TileTensor(c.buffer, c.layout)
    ctx.enqueue_function[kernel](
        ta.as_imm(),
        tb.as_imm(),
        tdummy.as_imm(),
        tc,
        scale,
        grid_dim=(ceildiv(N, TILE_SIZE), ceildiv(M, TILE_SIZE)),
        block_dim=(TILE_SIZE, TILE_SIZE),
    )


@inline(.always)
def gpu_matmul_nt_bias_row[
    dtype: DType, M: Int, N: Int, K: Int
](
    ctx: DeviceContext,
    a: DeviceMatrix[dtype, M, K],
    b: DeviceMatrix[dtype, N, K],
    bias: DeviceMatrix[dtype, N, 1],
    mut c: DeviceMatrix[dtype, M, N],
) raises:
    comptime kernel = matmul_nt_kernel[dtype, M, N, K, GPU_EPILOGUE_BIAS_ROW]
    var ta = TileTensor(a.buffer, a.layout)
    var tb = TileTensor(b.buffer, b.layout)
    var tbias = TileTensor(bias.buffer, bias.layout)
    var tc = TileTensor(c.buffer, c.layout)
    ctx.enqueue_function[kernel](
        ta.as_imm(),
        tb.as_imm(),
        tbias.as_imm(),
        tc,
        Scalar[dtype](1.0),
        grid_dim=(ceildiv(N, TILE_SIZE), ceildiv(M, TILE_SIZE)),
        block_dim=(TILE_SIZE, TILE_SIZE),
    )


@inline(.always)
def _launch_matmul_tn[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
    has_relu_mask: Bool,
    has_scale: Bool,
](
    ctx: DeviceContext,
    a: DeviceMatrix[dtype, K, M],
    b: DeviceMatrix[dtype, K, N],
    mask_buf: DeviceBuffer[dtype],
    mut c: DeviceMatrix[dtype, M, N],
    scale: Scalar[dtype] = 1.0,
) raises:
    comptime use_32 = (M >= 64 and N >= 64)
    var ta = TileTensor(a.buffer, a.layout)
    var tb = TileTensor(b.buffer, b.layout)
    var tm = TileTensor(mask_buf, row_major[M, N]())
    var tc = TileTensor(c.buffer, c.layout)
    comptime if use_32:
        comptime kernel = matmul_tn_32x32_kernel[
            dtype, M, N, K, has_relu_mask, has_scale
        ]
        ctx.enqueue_function[kernel](
            ta.as_imm(),
            tb.as_imm(),
            tm.as_imm(),
            tc,
            scale,
            grid_dim=(ceildiv(N, BIG_TILE), ceildiv(M, BIG_TILE)),
            block_dim=(TILE_SIZE, TILE_SIZE),
        )
    else:
        comptime kernel = matmul_tn_kernel[
            dtype, M, N, K, has_relu_mask, has_scale
        ]
        ctx.enqueue_function[kernel](
            ta.as_imm(),
            tb.as_imm(),
            tm.as_imm(),
            tc,
            scale,
            grid_dim=(ceildiv(N, TILE_SIZE), ceildiv(M, TILE_SIZE)),
            block_dim=(TILE_SIZE, TILE_SIZE),
        )


@inline(.always)
def gpu_matmul_tn[
    dtype: DType, M: Int, N: Int, K: Int
](
    ctx: DeviceContext,
    a: DeviceMatrix[dtype, K, M],
    b: DeviceMatrix[dtype, K, N],
    mut c: DeviceMatrix[dtype, M, N],
) raises:
    _launch_matmul_tn[dtype, M, N, K, False, False](ctx, a, b, a.buffer, c)


@inline(.always)
def gpu_matmul_tn_scaled[
    dtype: DType, M: Int, N: Int, K: Int
](
    ctx: DeviceContext,
    a: DeviceMatrix[dtype, K, M],
    b: DeviceMatrix[dtype, K, N],
    mut c: DeviceMatrix[dtype, M, N],
    scale: Scalar[dtype],
) raises:
    _launch_matmul_tn[dtype, M, N, K, False, True](
        ctx, a, b, a.buffer, c, scale
    )


@inline(.always)
def gpu_matmul_tn_relu_mask[
    dtype: DType, M: Int, N: Int, K: Int
](
    ctx: DeviceContext,
    a: DeviceMatrix[dtype, K, M],
    b: DeviceMatrix[dtype, K, N],
    mask_act: DeviceMatrix[dtype, M, N],
    mut c: DeviceMatrix[dtype, M, N],
) raises:
    _launch_matmul_tn[dtype, M, N, K, True, False](
        ctx, a, b, mask_act.buffer, c
    )


@inline(.always)
def gpu_add_bias_relu[
    dtype: DType, rows: Int, cols: Int
](
    ctx: DeviceContext,
    mut z: DeviceMatrix[dtype, rows, cols],
    bias: DeviceMatrix[dtype, rows, 1],
    mut a: DeviceMatrix[dtype, rows, cols],
) raises:
    comptime kernel = add_bias_relu_kernel[dtype, rows, cols]
    var tz = TileTensor(z.buffer, z.layout)
    var tb = TileTensor(bias.buffer, bias.layout)
    var ta = TileTensor(a.buffer, a.layout)
    ctx.enqueue_function[kernel](
        tz,
        tb.as_imm(),
        ta,
        grid_dim=(ceildiv(cols, TILE_SIZE), ceildiv(rows, TILE_SIZE)),
        block_dim=(TILE_SIZE, TILE_SIZE),
    )


@inline(.always)
def gpu_add_bias[
    dtype: DType, rows: Int, cols: Int
](
    ctx: DeviceContext,
    mut z: DeviceMatrix[dtype, rows, cols],
    bias: DeviceMatrix[dtype, rows, 1],
) raises:
    comptime kernel = add_bias_kernel[dtype, rows, cols]
    var tz = TileTensor(z.buffer, z.layout)
    var tb = TileTensor(bias.buffer, bias.layout)
    ctx.enqueue_function[kernel](
        tz,
        tb.as_imm(),
        grid_dim=(ceildiv(cols, TILE_SIZE), ceildiv(rows, TILE_SIZE)),
        block_dim=(TILE_SIZE, TILE_SIZE),
    )


@inline(.always)
def gpu_softmax[
    dtype: DType, rows: Int, cols: Int
](
    ctx: DeviceContext,
    z: DeviceMatrix[dtype, rows, cols],
    mut a: DeviceMatrix[dtype, rows, cols],
) raises:
    comptime kernel = softmax_col_kernel[dtype, rows, cols]
    comptime block = min(BLOCK_1D, align_up(cols, HOST_WARP_ALIGN))
    var tz = TileTensor(z.buffer, z.layout)
    var ta = TileTensor(a.buffer, a.layout)
    ctx.enqueue_function[kernel](
        tz.as_imm(),
        ta,
        grid_dim=ceildiv(cols, block),
        block_dim=block,
    )


def gpu_softmax_ce_grad[
    dtype: DType, rows: Int, cols: Int
](
    ctx: DeviceContext,
    a2: DeviceMatrix[dtype, rows, cols],
    labels: DeviceMatrix[DType.int32, cols, 1],
    mut dz2: DeviceMatrix[dtype, rows, cols],
    mut losses: DeviceMatrix[dtype, cols, 1],
) raises -> Scalar[dtype]:
    comptime kernel = softmax_ce_grad_kernel[dtype, rows, cols]
    comptime block = min(BLOCK_1D, align_up(cols, HOST_WARP_ALIGN))
    var ta = TileTensor(a2.buffer, a2.layout)
    var tl = TileTensor(labels.buffer, labels.layout)
    var td = TileTensor(dz2.buffer, dz2.layout)
    var tloss = TileTensor(losses.buffer, losses.layout)
    ctx.enqueue_function[kernel](
        ta.as_imm(),
        tl.as_imm(),
        td,
        tloss,
        grid_dim=ceildiv(cols, block),
        block_dim=block,
    )
    var total_loss: Scalar[dtype] = 0.0
    with losses.buffer.map_to_host() as host_loss:
        for i in range(cols):
            total_loss += host_loss[i]
    return total_loss / Scalar[dtype](cols)


@inline(.always)
def gpu_relu_backward_mask[
    dtype: DType, rows: Int, cols: Int
](
    ctx: DeviceContext,
    z: DeviceMatrix[dtype, rows, cols],
    mut dz: DeviceMatrix[dtype, rows, cols],
) raises:
    comptime kernel = relu_backward_mask_kernel[dtype, rows, cols]
    comptime total = rows * cols
    comptime block = min(BLOCK_1D, align_up(total, HOST_WARP_ALIGN))
    var tz = TileTensor(z.buffer, z.layout)
    var td = TileTensor(dz.buffer, dz.layout)
    ctx.enqueue_function[kernel](
        tz.as_imm(),
        td,
        grid_dim=ceildiv(total, block),
        block_dim=block,
    )


@inline(.always)
def gpu_scale[
    dtype: DType, rows: Int, cols: Int
](
    ctx: DeviceContext,
    mut m: DeviceMatrix[dtype, rows, cols],
    scale: Scalar[dtype],
) raises:
    comptime kernel = scale_kernel[dtype, rows, cols]
    comptime total = rows * cols
    comptime block = min(BLOCK_1D, align_up(total, HOST_WARP_ALIGN))
    var tm = TileTensor(m.buffer, m.layout)
    ctx.enqueue_function[kernel](
        tm,
        scale,
        grid_dim=ceildiv(total, block),
        block_dim=block,
    )


@inline(.always)
def gpu_sum_axis1_scaled[
    dtype: DType, rows: Int, cols: Int
](
    ctx: DeviceContext,
    m: DeviceMatrix[dtype, rows, cols],
    scale: Scalar[dtype],
    mut out_v: DeviceMatrix[dtype, rows, 1],
) raises:
    comptime block_threads = min(BLOCK_1D, align_up(cols, HOST_WARP_ALIGN))
    comptime kernel = sum_axis1_scaled_kernel[dtype, rows, cols, block_threads]
    var tm = TileTensor(m.buffer, m.layout)
    var tout = TileTensor(out_v.buffer, out_v.layout)
    ctx.enqueue_function[kernel](
        tm.as_imm(),
        tout,
        scale,
        grid_dim=rows,
        block_dim=block_threads,
    )


@inline(.always)
def gpu_sgd_momentum_update[
    dtype: DType, rows: Int, cols: Int
](
    ctx: DeviceContext,
    mut param: DeviceMatrix[dtype, rows, cols],
    grad: DeviceMatrix[dtype, rows, cols],
    mut vel: DeviceMatrix[dtype, rows, cols],
    lr: Scalar[dtype],
    momentum: Scalar[dtype],
) raises:
    comptime kernel = sgd_momentum_kernel[dtype, rows, cols]
    comptime total = rows * cols
    comptime block = min(BLOCK_1D, align_up(total, HOST_WARP_ALIGN))
    var tp = TileTensor(param.buffer, param.layout)
    var tg = TileTensor(grad.buffer, grad.layout)
    var tv = TileTensor(vel.buffer, vel.layout)
    ctx.enqueue_function[kernel](
        tp,
        tg.as_imm(),
        tv,
        lr,
        momentum,
        grid_dim=ceildiv(total, block),
        block_dim=block,
    )


@inline(.always)
def gpu_argmax_axis0[
    dtype: DType, rows: Int, cols: Int
](
    ctx: DeviceContext,
    probs: DeviceMatrix[dtype, rows, cols],
    mut preds: DeviceMatrix[DType.int32, cols, 1],
) raises:
    comptime kernel = argmax_axis0_kernel[dtype, rows, cols]
    comptime block = min(BLOCK_1D, align_up(cols, HOST_WARP_ALIGN))
    var tp = TileTensor(probs.buffer, probs.layout)
    var tout = TileTensor(preds.buffer, preds.layout)
    ctx.enqueue_function[kernel](
        tp.as_imm(),
        tout,
        grid_dim=ceildiv(cols, block),
        block_dim=block,
    )


def gather_batch_cols_kernel[
    dtype: DType, total_samples: Int, num_features: Int, batch_size: Int
](
    x_all: ImmTileTensor[
        dtype,
        type_of(row_major[total_samples, num_features]()),
        ImmUntrackedOrigin,
    ],
    y_all: ImmTileTensor[
        DType.int32, type_of(row_major[total_samples, 1]()), ImmUntrackedOrigin
    ],
    indices: ImmTileTensor[
        DType.int32, type_of(row_major[total_samples, 1]()), ImmUntrackedOrigin
    ],
    batch_start: Int32,
    x_batch: MutTileTensor[
        dtype,
        type_of(row_major[num_features, batch_size]()),
        MutUntrackedOrigin,
    ],
    y_batch: MutTileTensor[
        DType.int32, type_of(row_major[batch_size, 1]()), MutUntrackedOrigin
    ],
):
    """Gathers shuffled mini-batch samples on GPU into (num_features, batch_size).
    """
    comptime assert (
        x_all.flat_rank == 2
        and y_all.flat_rank == 2
        and indices.flat_rank == 2
        and x_batch.flat_rank == 2
        and y_batch.flat_rank == 2
    )
    var col = global_idx.x
    var feat = global_idx.y
    if col < batch_size and feat < num_features:
        var sample_idx = Int(indices[Int(batch_start) + col, 0])
        x_batch[feat, col] = rebind[x_batch.ElementType](
            x_all[sample_idx, feat]
        )
        if feat == 0:
            y_batch[col, 0] = y_all[sample_idx, 0]


def gather_eval_cols_kernel[
    dtype: DType, total_samples: Int, num_features: Int, batch_size: Int
](
    x_all: ImmTileTensor[
        dtype,
        type_of(row_major[total_samples, num_features]()),
        ImmUntrackedOrigin,
    ],
    start_idx: Int32,
    x_batch: MutTileTensor[
        dtype,
        type_of(row_major[num_features, batch_size]()),
        MutUntrackedOrigin,
    ],
):
    """Gathers contiguous eval samples on GPU into (num_features, batch_size).
    """
    comptime assert x_all.flat_rank == 2 and x_batch.flat_rank == 2
    var col = global_idx.x
    var feat = global_idx.y
    if col < batch_size and feat < num_features:
        var sample_idx = Int(start_idx) + col
        x_batch[feat, col] = rebind[x_batch.ElementType](
            x_all[sample_idx, feat]
        )


def softmax_ce_grad_metrics_kernel[
    dtype: DType, num_classes: Int, batch_size: Int, block_threads: Int
](
    z2: ImmTileTensor[
        dtype, type_of(row_major[num_classes, batch_size]()), ImmUntrackedOrigin
    ],
    labels: ImmTileTensor[
        DType.int32, type_of(row_major[batch_size, 1]()), ImmUntrackedOrigin
    ],
    dz2: MutTileTensor[
        dtype, type_of(row_major[num_classes, batch_size]()), MutUntrackedOrigin
    ],
    epoch_loss: MutTileTensor[
        dtype, type_of(row_major[1, 1]()), MutUntrackedOrigin
    ],
    epoch_correct: MutTileTensor[
        DType.int32, type_of(row_major[1, 1]()), MutUntrackedOrigin
    ],
):
    """Fused softmax, CE loss, gradient dZ2, and correct count in registers + warp reductions.
    """
    comptime assert (
        dtype.is_floating_point()
    ), "cross-entropy requires floating-point dtype"
    comptime assert (
        z2.flat_rank == 2
        and labels.flat_rank == 2
        and dz2.flat_rank == 2
        and epoch_loss.flat_rank == 2
        and epoch_correct.flat_rank == 2
    )
    comptime num_warps = block_threads // WARP_SIZE
    comptime inv_batch = Scalar[dtype](1.0) / Scalar[dtype](batch_size)

    var tid = thread_idx.x
    var my_loss: Scalar[dtype] = 0.0
    var my_correct: Int32 = 0

    if tid < batch_size:
        var lbl = Int(labels[tid, 0])
        var vals = Array[Scalar[dtype], num_classes](uninitialized=True)
        vals[0] = rebind[Scalar[dtype]](z2[0, tid])
        var col_max = vals[0]
        var best_r = 0
        comptime for r in range(1, num_classes):
            var v = rebind[Scalar[dtype]](z2[r, tid])
            vals[r] = v
            if v > col_max:
                col_max = v
                best_r = r

        var sum_exp: Scalar[dtype] = 0.0
        comptime for r in range(num_classes):
            var e_val = exp(vals[r] - col_max)
            vals[r] = e_val
            sum_exp += e_val

        var inv_sum = Scalar[dtype](1.0) / sum_exp
        comptime for r in range(num_classes):
            var p = vals[r] * inv_sum
            if r == lbl:
                dz2[r, tid] = rebind[dz2.ElementType](p - Scalar[dtype](1.0))
                my_loss = -log(max(p, Scalar[dtype](1e-7)))
            else:
                dz2[r, tid] = rebind[dz2.ElementType](p)

        if best_r == lbl:
            my_correct = 1

    var warp_loss = warp.sum(my_loss)
    var warp_correct = warp.sum(my_correct)

    comptime if num_warps <= 1:
        if tid == 0:
            epoch_loss[0, 0] = epoch_loss[0, 0] + rebind[
                epoch_loss.ElementType
            ](warp_loss * inv_batch)
            epoch_correct[0, 0] = epoch_correct[0, 0] + warp_correct
    else:
        var s_loss = stack_allocation[dtype, address_space=.SHARED](
            row_major[num_warps]()
        )
        var s_correct = stack_allocation[DType.int32, address_space=.SHARED](
            row_major[num_warps]()
        )
        var wid = tid // WARP_SIZE
        if lane_id() == 0:
            s_loss[wid] = rebind[s_loss.ElementType](warp_loss)
            s_correct[wid] = warp_correct
        barrier()

        if tid == 0:
            var total_loss: Scalar[dtype] = 0.0
            var total_correct: Int32 = 0
            comptime for w in range(num_warps):
                total_loss += rebind[Scalar[dtype]](s_loss[w])
                total_correct += rebind[Int32](s_correct[w])
            epoch_loss[0, 0] = epoch_loss[0, 0] + rebind[
                epoch_loss.ElementType
            ](total_loss * inv_batch)
            epoch_correct[0, 0] = epoch_correct[0, 0] + total_correct


def batch_ce_grad_metrics_kernel[
    dtype: DType, num_classes: Int, batch_size: Int, block_threads: Int
](
    a2: ImmTileTensor[
        dtype, type_of(row_major[num_classes, batch_size]()), ImmUntrackedOrigin
    ],
    labels: ImmTileTensor[
        DType.int32, type_of(row_major[batch_size, 1]()), ImmUntrackedOrigin
    ],
    dz2: MutTileTensor[
        dtype, type_of(row_major[num_classes, batch_size]()), MutUntrackedOrigin
    ],
    epoch_loss: MutTileTensor[
        dtype, type_of(row_major[1, 1]()), MutUntrackedOrigin
    ],
    epoch_correct: MutTileTensor[
        DType.int32, type_of(row_major[1, 1]()), MutUntrackedOrigin
    ],
):
    """Fused dZ2 = A2 - one_hot(labels), batch loss, and batch correct count."""
    comptime assert (
        dtype.is_floating_point()
    ), "cross-entropy requires floating-point dtype"
    comptime assert (
        a2.flat_rank == 2
        and labels.flat_rank == 2
        and dz2.flat_rank == 2
        and epoch_loss.flat_rank == 2
        and epoch_correct.flat_rank == 2
    )
    comptime num_warps = block_threads // WARP_SIZE
    comptime inv_batch = Scalar[dtype](1.0) / Scalar[dtype](batch_size)

    var tid = thread_idx.x
    var my_loss: Scalar[dtype] = 0.0
    var my_correct: Int32 = 0

    if tid < batch_size:
        var lbl = Int(labels[tid, 0])
        var best_r = 0
        var best_v = rebind[Scalar[dtype]](a2[0, tid])

        comptime for r in range(num_classes):
            var p = rebind[Scalar[dtype]](a2[r, tid])
            if p > best_v:
                best_v = p
                best_r = r
            if r == lbl:
                dz2[r, tid] = rebind[dz2.ElementType](p - Scalar[dtype](1.0))
                my_loss = -log(max(p, Scalar[dtype](1e-7)))
            else:
                dz2[r, tid] = rebind[dz2.ElementType](p)

        if best_r == lbl:
            my_correct = 1

    var warp_loss = warp.sum(my_loss)
    var warp_correct = warp.sum(my_correct)

    comptime if num_warps <= 1:
        if tid == 0:
            epoch_loss[0, 0] = epoch_loss[0, 0] + rebind[
                epoch_loss.ElementType
            ](warp_loss * inv_batch)
            epoch_correct[0, 0] = epoch_correct[0, 0] + warp_correct
    else:
        var s_loss = stack_allocation[dtype, address_space=.SHARED](
            row_major[num_warps]()
        )
        var s_correct = stack_allocation[DType.int32, address_space=.SHARED](
            row_major[num_warps]()
        )
        var wid = tid // WARP_SIZE
        if lane_id() == 0:
            s_loss[wid] = rebind[s_loss.ElementType](warp_loss)
            s_correct[wid] = warp_correct
        barrier()

        if tid == 0:
            var total_loss: Scalar[dtype] = 0.0
            var total_correct: Int32 = 0
            comptime for w in range(num_warps):
                total_loss += rebind[Scalar[dtype]](s_loss[w])
                total_correct += rebind[Int32](s_correct[w])
            epoch_loss[0, 0] = epoch_loss[0, 0] + rebind[
                epoch_loss.ElementType
            ](total_loss * inv_batch)
            epoch_correct[0, 0] = epoch_correct[0, 0] + total_correct


def eval_argmax_count_kernel[
    dtype: DType,
    num_classes: Int,
    batch_size: Int,
    total_samples: Int,
    has_labels: Bool,
    block_threads: Int,
](
    probs: ImmTileTensor[
        dtype, type_of(row_major[num_classes, batch_size]()), ImmUntrackedOrigin
    ],
    labels: ImmTileTensor[
        DType.int32, type_of(row_major[total_samples, 1]()), ImmUntrackedOrigin
    ],
    offset: Int32,
    out_preds: MutTileTensor[
        DType.int32, type_of(row_major[total_samples, 1]()), MutUntrackedOrigin
    ],
    total_correct: MutTileTensor[
        DType.int32, type_of(row_major[1, 1]()), MutUntrackedOrigin
    ],
):
    """Per-batch argmax and optional accuracy accumulation on GPU."""
    comptime assert (
        probs.flat_rank == 2
        and labels.flat_rank == 2
        and out_preds.flat_rank == 2
        and total_correct.flat_rank == 2
    )
    comptime num_warps = block_threads // WARP_SIZE

    var tid = thread_idx.x
    var my_correct: Int32 = 0

    if tid < batch_size:
        var best_r = Int32(0)
        var best_v = rebind[Scalar[dtype]](probs[0, tid])
        comptime for r in range(1, num_classes):
            var v = rebind[Scalar[dtype]](probs[r, tid])
            if v > best_v:
                best_v = v
                best_r = Int32(r)

        var global_sample = Int(offset) + tid
        out_preds[global_sample, 0] = best_r

        comptime if has_labels:
            if best_r == labels[global_sample, 0]:
                my_correct = 1

    comptime if has_labels:
        var warp_correct = warp.sum(my_correct)
        comptime if num_warps <= 1:
            if tid == 0:
                total_correct[0, 0] = total_correct[0, 0] + warp_correct
        else:
            var s_correct = stack_allocation[
                DType.int32, address_space=.SHARED
            ](row_major[num_warps]())
            var wid = tid // WARP_SIZE
            if lane_id() == 0:
                s_correct[wid] = warp_correct
            barrier()
            if tid == 0:
                var sum_c: Int32 = 0
                comptime for w in range(num_warps):
                    sum_c += rebind[Int32](s_correct[w])
                total_correct[0, 0] = total_correct[0, 0] + sum_c


@inline(.always)
def gpu_gather_batch_cols[
    dtype: DType, total_samples: Int, num_features: Int, batch_size: Int
](
    ctx: DeviceContext,
    x_all: DeviceMatrix[dtype, total_samples, num_features],
    y_all: DeviceMatrix[DType.int32, total_samples, 1],
    indices: DeviceMatrix[DType.int32, total_samples, 1],
    batch_start: Int,
    mut x_batch: DeviceMatrix[dtype, num_features, batch_size],
    mut y_batch: DeviceMatrix[DType.int32, batch_size, 1],
) raises:
    comptime kernel = gather_batch_cols_kernel[
        dtype, total_samples, num_features, batch_size
    ]
    var tx = TileTensor(x_all.buffer, x_all.layout)
    var ty = TileTensor(y_all.buffer, y_all.layout)
    var tidx = TileTensor(indices.buffer, indices.layout)
    var txb = TileTensor(x_batch.buffer, x_batch.layout)
    var tyb = TileTensor(y_batch.buffer, y_batch.layout)
    ctx.enqueue_function[kernel](
        tx.as_imm(),
        ty.as_imm(),
        tidx.as_imm(),
        Int32(batch_start),
        txb,
        tyb,
        grid_dim=(
            ceildiv(batch_size, TILE_SIZE),
            ceildiv(num_features, TILE_SIZE),
        ),
        block_dim=(TILE_SIZE, TILE_SIZE),
    )


@inline(.always)
def gpu_gather_eval_cols[
    dtype: DType, total_samples: Int, num_features: Int, batch_size: Int
](
    ctx: DeviceContext,
    x_all: DeviceMatrix[dtype, total_samples, num_features],
    start_idx: Int,
    mut x_batch: DeviceMatrix[dtype, num_features, batch_size],
) raises:
    comptime kernel = gather_eval_cols_kernel[
        dtype, total_samples, num_features, batch_size
    ]
    var tx = TileTensor(x_all.buffer, x_all.layout)
    var txb = TileTensor(x_batch.buffer, x_batch.layout)
    ctx.enqueue_function[kernel](
        tx.as_imm(),
        Int32(start_idx),
        txb,
        grid_dim=(
            ceildiv(batch_size, TILE_SIZE),
            ceildiv(num_features, TILE_SIZE),
        ),
        block_dim=(TILE_SIZE, TILE_SIZE),
    )


@inline(.always)
def gpu_softmax_ce_grad_metrics[
    dtype: DType, num_classes: Int, batch_size: Int
](
    ctx: DeviceContext,
    z2: DeviceMatrix[dtype, num_classes, batch_size],
    labels: DeviceMatrix[DType.int32, batch_size, 1],
    mut dz2: DeviceMatrix[dtype, num_classes, batch_size],
    mut epoch_loss: DeviceMatrix[dtype, 1, 1],
    mut epoch_correct: DeviceMatrix[DType.int32, 1, 1],
) raises:
    comptime block_threads = min(
        BLOCK_1D, align_up(batch_size, HOST_WARP_ALIGN)
    )
    comptime kernel = softmax_ce_grad_metrics_kernel[
        dtype, num_classes, batch_size, block_threads
    ]
    var tz = TileTensor(z2.buffer, z2.layout)
    var tl = TileTensor(labels.buffer, labels.layout)
    var td = TileTensor(dz2.buffer, dz2.layout)
    var tel = TileTensor(epoch_loss.buffer, epoch_loss.layout)
    var tec = TileTensor(epoch_correct.buffer, epoch_correct.layout)
    ctx.enqueue_function[kernel](
        tz.as_imm(),
        tl.as_imm(),
        td,
        tel,
        tec,
        grid_dim=1,
        block_dim=block_threads,
    )


@inline(.always)
def gpu_batch_ce_grad_metrics[
    dtype: DType, num_classes: Int, batch_size: Int
](
    ctx: DeviceContext,
    a2: DeviceMatrix[dtype, num_classes, batch_size],
    labels: DeviceMatrix[DType.int32, batch_size, 1],
    mut dz2: DeviceMatrix[dtype, num_classes, batch_size],
    mut epoch_loss: DeviceMatrix[dtype, 1, 1],
    mut epoch_correct: DeviceMatrix[DType.int32, 1, 1],
) raises:
    comptime block_threads = min(
        BLOCK_1D, align_up(batch_size, HOST_WARP_ALIGN)
    )
    comptime kernel = batch_ce_grad_metrics_kernel[
        dtype, num_classes, batch_size, block_threads
    ]
    var ta = TileTensor(a2.buffer, a2.layout)
    var tl = TileTensor(labels.buffer, labels.layout)
    var td = TileTensor(dz2.buffer, dz2.layout)
    var tel = TileTensor(epoch_loss.buffer, epoch_loss.layout)
    var tec = TileTensor(epoch_correct.buffer, epoch_correct.layout)
    ctx.enqueue_function[kernel](
        ta.as_imm(),
        tl.as_imm(),
        td,
        tel,
        tec,
        grid_dim=1,
        block_dim=block_threads,
    )


@inline(.always)
def gpu_eval_argmax_count[
    dtype: DType,
    num_classes: Int,
    batch_size: Int,
    total_samples: Int,
    has_labels: Bool,
](
    ctx: DeviceContext,
    probs: DeviceMatrix[dtype, num_classes, batch_size],
    labels: DeviceMatrix[DType.int32, total_samples, 1],
    offset: Int,
    mut out_preds: DeviceMatrix[DType.int32, total_samples, 1],
    mut total_correct: DeviceMatrix[DType.int32, 1, 1],
) raises:
    comptime block_threads = min(
        BLOCK_1D, align_up(batch_size, HOST_WARP_ALIGN)
    )
    comptime kernel = eval_argmax_count_kernel[
        dtype, num_classes, batch_size, total_samples, has_labels, block_threads
    ]
    var tp = TileTensor(probs.buffer, probs.layout)
    var tl = TileTensor(labels.buffer, labels.layout)
    var top = TileTensor(out_preds.buffer, out_preds.layout)
    var tc = TileTensor(total_correct.buffer, total_correct.layout)
    ctx.enqueue_function[kernel](
        tp.as_imm(),
        tl.as_imm(),
        Int32(offset),
        top,
        tc,
        grid_dim=1,
        block_dim=block_threads,
    )


def sum_axis0_scaled_kernel[
    dtype: DType, rows: Int, cols: Int
](
    m: ImmTileTensor[
        dtype, type_of(row_major[rows, cols]()), ImmUntrackedOrigin
    ],
    scale: Scalar[dtype],
    out_v: MutTileTensor[
        dtype, type_of(row_major[cols, 1]()), MutUntrackedOrigin
    ],
):
    """Coalesced column-sum: out_v[col, 0] = scale * sum_r m[r, col]."""
    comptime assert m.flat_rank == 2 and out_v.flat_rank == 2
    var col = global_idx.x
    if col >= cols:
        return

    var acc: Scalar[dtype] = 0.0
    for r in range(rows):
        acc += rebind[Scalar[dtype]](m[r, col])
    out_v[col, 0] = rebind[out_v.ElementType](acc * scale)


@inline(.always)
def gpu_sum_axis0_scaled[
    dtype: DType, rows: Int, cols: Int
](
    ctx: DeviceContext,
    m: DeviceMatrix[dtype, rows, cols],
    scale: Scalar[dtype],
    mut out_v: DeviceMatrix[dtype, cols, 1],
) raises:
    comptime block_threads = min(BLOCK_1D, align_up(cols, HOST_WARP_ALIGN))
    comptime kernel = sum_axis0_scaled_kernel[dtype, rows, cols]
    var tm = TileTensor(m.buffer, m.layout)
    var to = TileTensor(out_v.buffer, out_v.layout)
    ctx.enqueue_function[kernel](
        tm.as_imm(),
        scale,
        to,
        grid_dim=ceildiv(cols, block_threads),
        block_dim=block_threads,
    )


def gather_batch_rows_kernel[
    dtype: DType, total_samples: Int, num_features: Int, batch_size: Int
](
    x_all: ImmTileTensor[
        dtype,
        type_of(row_major[total_samples, num_features]()),
        ImmUntrackedOrigin,
    ],
    y_all: ImmTileTensor[
        DType.int32, type_of(row_major[total_samples, 1]()), ImmUntrackedOrigin
    ],
    indices: ImmTileTensor[
        DType.int32, type_of(row_major[total_samples, 1]()), ImmUntrackedOrigin
    ],
    batch_start: Int32,
    x_batch: MutTileTensor[
        dtype,
        type_of(row_major[batch_size, num_features]()),
        MutUntrackedOrigin,
    ],
    y_batch: MutTileTensor[
        DType.int32, type_of(row_major[batch_size, 1]()), MutUntrackedOrigin
    ],
):
    """Coalesced row-gather into sample-major batch (batch_size, num_features).
    """
    comptime assert (
        x_all.flat_rank == 2
        and y_all.flat_rank == 2
        and indices.flat_rank == 2
        and x_batch.flat_rank == 2
        and y_batch.flat_rank == 2
    )
    var b = block_idx.x
    var tid = thread_idx.x
    if b >= batch_size:
        return

    var src_row = Int(indices[Int(batch_start) + b, 0])
    if tid == 0:
        y_batch[b, 0] = y_all[src_row, 0]

    for feat in range(tid, num_features, BLOCK_1D):
        x_batch[b, feat] = rebind[x_batch.ElementType](x_all[src_row, feat])


@inline(.always)
def gpu_gather_batch_rows[
    dtype: DType, total_samples: Int, num_features: Int, batch_size: Int
](
    ctx: DeviceContext,
    x_all: DeviceMatrix[dtype, total_samples, num_features],
    y_all: DeviceMatrix[DType.int32, total_samples, 1],
    indices: DeviceMatrix[DType.int32, total_samples, 1],
    batch_start: Int,
    mut x_batch: DeviceMatrix[dtype, batch_size, num_features],
    mut y_batch: DeviceMatrix[DType.int32, batch_size, 1],
) raises:
    comptime kernel = gather_batch_rows_kernel[
        dtype, total_samples, num_features, batch_size
    ]
    var tx = TileTensor(x_all.buffer, x_all.layout)
    var ty = TileTensor(y_all.buffer, y_all.layout)
    var tidx = TileTensor(indices.buffer, indices.layout)
    var txb = TileTensor(x_batch.buffer, x_batch.layout)
    var tyb = TileTensor(y_batch.buffer, y_batch.layout)
    ctx.enqueue_function[kernel](
        tx.as_imm(),
        ty.as_imm(),
        tidx.as_imm(),
        Int32(batch_start),
        txb,
        tyb,
        grid_dim=batch_size,
        block_dim=BLOCK_1D,
    )


def softmax_ce_grad_metrics_db_rows_kernel[
    dtype: DType, batch_size: Int, num_classes: Int, block_threads: Int
](
    z2: ImmTileTensor[
        dtype, type_of(row_major[batch_size, num_classes]()), ImmUntrackedOrigin
    ],
    labels: ImmTileTensor[
        DType.int32, type_of(row_major[batch_size, 1]()), ImmUntrackedOrigin
    ],
    dz2: MutTileTensor[
        dtype, type_of(row_major[batch_size, num_classes]()), MutUntrackedOrigin
    ],
    db2: MutTileTensor[
        dtype, type_of(row_major[num_classes, 1]()), MutUntrackedOrigin
    ],
    epoch_loss: MutTileTensor[
        dtype, type_of(row_major[1, 1]()), MutUntrackedOrigin
    ],
    epoch_correct: MutTileTensor[
        DType.int32, type_of(row_major[1, 1]()), MutUntrackedOrigin
    ],
):
    """Fused sample-major softmax, CE loss, dZ2, dB2, and batch accuracy."""
    comptime assert dtype.is_floating_point()
    comptime assert (
        z2.flat_rank == 2
        and labels.flat_rank == 2
        and dz2.flat_rank == 2
        and db2.flat_rank == 2
        and epoch_loss.flat_rank == 2
        and epoch_correct.flat_rank == 2
    )
    comptime num_warps = block_threads // WARP_SIZE
    comptime inv_batch = Scalar[dtype](1.0) / Scalar[dtype](batch_size)

    var tid = thread_idx.x
    var my_loss: Scalar[dtype] = 0.0
    var my_correct: Int32 = 0
    var grads = Array[Scalar[dtype], num_classes](fill=0.0)

    if tid < batch_size:
        var lbl = Int(labels[tid, 0])
        grads[0] = rebind[Scalar[dtype]](z2[tid, 0])
        var row_max = grads[0]
        var best_c = 0
        comptime for c in range(1, num_classes):
            var v = rebind[Scalar[dtype]](z2[tid, c])
            grads[c] = v
            if v > row_max:
                row_max = v
                best_c = c

        var sum_exp: Scalar[dtype] = 0.0
        comptime for c in range(num_classes):
            var e_val = exp(grads[c] - row_max)
            grads[c] = e_val
            sum_exp += e_val

        var inv_sum = Scalar[dtype](1.0) / sum_exp
        comptime for c in range(num_classes):
            var p = grads[c] * inv_sum
            var g = p - Scalar[dtype](1.0) if c == lbl else p
            if c == lbl:
                my_loss = -log(max(p, Scalar[dtype](1e-7)))
            grads[c] = g
            dz2[tid, c] = rebind[dz2.ElementType](g)

        if best_c == lbl:
            my_correct = 1

    var warp_loss = warp.sum(my_loss)
    var warp_correct = warp.sum(my_correct)

    comptime if num_warps <= 1:
        comptime for c in range(num_classes):
            var w_db = warp.sum(grads[c])
            if tid == 0:
                db2[c, 0] = rebind[db2.ElementType](w_db * inv_batch)
        if tid == 0:
            epoch_loss[0, 0] = epoch_loss[0, 0] + rebind[
                epoch_loss.ElementType
            ](warp_loss * inv_batch)
            epoch_correct[0, 0] = epoch_correct[0, 0] + warp_correct
    else:
        var s_loss = stack_allocation[dtype, address_space=.SHARED](
            row_major[num_warps]()
        )
        var s_correct = stack_allocation[DType.int32, address_space=.SHARED](
            row_major[num_warps]()
        )
        var s_db2 = stack_allocation[dtype, address_space=.SHARED](
            row_major[num_warps, num_classes]()
        )
        var wid = tid // WARP_SIZE
        var lane = lane_id()
        if lane == 0:
            s_loss[wid] = rebind[s_loss.ElementType](warp_loss)
            s_correct[wid] = warp_correct
        comptime for c in range(num_classes):
            var w_db = warp.sum(grads[c])
            if lane == 0:
                s_db2[wid, c] = rebind[s_db2.ElementType](w_db)
        barrier()

        if tid == 0:
            var total_loss: Scalar[dtype] = 0.0
            var total_correct: Int32 = 0
            comptime for w in range(num_warps):
                total_loss += rebind[Scalar[dtype]](s_loss[w])
                total_correct += rebind[Int32](s_correct[w])
            epoch_loss[0, 0] = epoch_loss[0, 0] + rebind[
                epoch_loss.ElementType
            ](total_loss * inv_batch)
            epoch_correct[0, 0] = epoch_correct[0, 0] + total_correct

        if tid < num_classes:
            var total_db: Scalar[dtype] = 0.0
            comptime for w in range(num_warps):
                total_db += rebind[Scalar[dtype]](s_db2[w, tid])
            db2[tid, 0] = rebind[db2.ElementType](total_db * inv_batch)


@inline(.always)
def gpu_softmax_ce_grad_metrics_db_rows[
    dtype: DType, batch_size: Int, num_classes: Int
](
    ctx: DeviceContext,
    z2: DeviceMatrix[dtype, batch_size, num_classes],
    labels: DeviceMatrix[DType.int32, batch_size, 1],
    mut dz2: DeviceMatrix[dtype, batch_size, num_classes],
    mut db2: DeviceMatrix[dtype, num_classes, 1],
    mut epoch_loss: DeviceMatrix[dtype, 1, 1],
    mut epoch_correct: DeviceMatrix[DType.int32, 1, 1],
) raises:
    comptime block_threads = min(
        BLOCK_1D, align_up(batch_size, HOST_WARP_ALIGN)
    )
    comptime kernel = softmax_ce_grad_metrics_db_rows_kernel[
        dtype, batch_size, num_classes, block_threads
    ]
    var tz = TileTensor(z2.buffer, z2.layout)
    var tl = TileTensor(labels.buffer, labels.layout)
    var td = TileTensor(dz2.buffer, dz2.layout)
    var tdb = TileTensor(db2.buffer, db2.layout)
    var tel = TileTensor(epoch_loss.buffer, epoch_loss.layout)
    var tec = TileTensor(epoch_correct.buffer, epoch_correct.layout)
    ctx.enqueue_function[kernel](
        tz.as_imm(),
        tl.as_imm(),
        td,
        tdb,
        tel,
        tec,
        grid_dim=1,
        block_dim=block_threads,
    )


def eval_argmax_rows_count_kernel[
    dtype: DType,
    batch_size: Int,
    num_classes: Int,
    total_samples: Int,
    has_labels: Bool,
    block_threads: Int,
](
    z2: ImmTileTensor[
        dtype, type_of(row_major[batch_size, num_classes]()), ImmUntrackedOrigin
    ],
    labels: ImmTileTensor[
        DType.int32, type_of(row_major[total_samples, 1]()), ImmUntrackedOrigin
    ],
    offset: Int32,
    out_preds: MutTileTensor[
        DType.int32, type_of(row_major[total_samples, 1]()), MutUntrackedOrigin
    ],
    total_correct: MutTileTensor[
        DType.int32, type_of(row_major[1, 1]()), MutUntrackedOrigin
    ],
):
    """Strided sample-major argmax and accuracy accumulation across eval batches.
    """
    comptime assert (
        z2.flat_rank == 2
        and labels.flat_rank == 2
        and out_preds.flat_rank == 2
        and total_correct.flat_rank == 2
    )
    comptime num_warps = block_threads // WARP_SIZE

    var tid = thread_idx.x
    var my_correct: Int32 = 0

    for b in range(tid, batch_size, block_threads):
        var best_c = Int32(0)
        var best_v = rebind[Scalar[dtype]](z2[b, 0])
        comptime for c in range(1, num_classes):
            var v = rebind[Scalar[dtype]](z2[b, c])
            if v > best_v:
                best_v = v
                best_c = Int32(c)

        var global_sample = Int(offset) + b
        out_preds[global_sample, 0] = best_c

        comptime if has_labels:
            if best_c == labels[global_sample, 0]:
                my_correct += 1

    comptime if has_labels:
        var warp_correct = warp.sum(my_correct)
        comptime if num_warps <= 1:
            if tid == 0:
                total_correct[0, 0] = total_correct[0, 0] + warp_correct
        else:
            var s_correct = stack_allocation[
                DType.int32, address_space=.SHARED
            ](row_major[num_warps]())
            var wid = tid // WARP_SIZE
            if lane_id() == 0:
                s_correct[wid] = warp_correct
            barrier()
            if tid == 0:
                var sum_c: Int32 = 0
                comptime for w in range(num_warps):
                    sum_c += rebind[Int32](s_correct[w])
                total_correct[0, 0] = total_correct[0, 0] + sum_c


@inline(.always)
def gpu_eval_argmax_rows_count[
    dtype: DType,
    batch_size: Int,
    num_classes: Int,
    total_samples: Int,
    has_labels: Bool,
](
    ctx: DeviceContext,
    z2: DeviceMatrix[dtype, batch_size, num_classes],
    labels: DeviceMatrix[DType.int32, total_samples, 1],
    offset: Int,
    mut out_preds: DeviceMatrix[DType.int32, total_samples, 1],
    mut total_correct: DeviceMatrix[DType.int32, 1, 1],
) raises:
    comptime block_threads = min(
        BLOCK_1D, align_up(batch_size, HOST_WARP_ALIGN)
    )
    comptime kernel = eval_argmax_rows_count_kernel[
        dtype,
        batch_size,
        num_classes,
        total_samples,
        has_labels,
        block_threads,
    ]
    var tz = TileTensor(z2.buffer, z2.layout)
    var tl = TileTensor(labels.buffer, labels.layout)
    var top = TileTensor(out_preds.buffer, out_preds.layout)
    var tc = TileTensor(total_correct.buffer, total_correct.layout)
    ctx.enqueue_function[kernel](
        tz.as_imm(),
        tl.as_imm(),
        Int32(offset),
        top,
        tc,
        grid_dim=1,
        block_dim=block_threads,
    )


def gather_batch_rows_shifted_kernel[
    dtype: DType, total_samples: Int, num_features: Int, batch_size: Int
](
    x_all: ImmTileTensor[
        dtype,
        type_of(row_major[total_samples, num_features]()),
        ImmUntrackedOrigin,
    ],
    y_all: ImmTileTensor[
        DType.int32, type_of(row_major[total_samples, 1]()), ImmUntrackedOrigin
    ],
    indices: ImmTileTensor[
        DType.int32, type_of(row_major[total_samples, 1]()), ImmUntrackedOrigin
    ],
    batch_start: Int32,
    seed_val: UInt32,
    x_batch: MutTileTensor[
        dtype,
        type_of(row_major[batch_size, num_features]()),
        MutUntrackedOrigin,
    ],
    y_batch: MutTileTensor[
        DType.int32, type_of(row_major[batch_size, 1]()), MutUntrackedOrigin
    ],
):
    """Coalesced 28x28 row gather with per-sample +/-1 pixel shift augmentation.
    """
    comptime assert (
        x_all.flat_rank == 2
        and y_all.flat_rank == 2
        and indices.flat_rank == 2
        and x_batch.flat_rank == 2
        and y_batch.flat_rank == 2
    )
    var b = block_idx.x
    var tid = thread_idx.x
    if b >= batch_size:
        return

    var src_row = Int(indices[Int(batch_start) + b, 0])
    if tid == 0:
        y_batch[b, 0] = y_all[src_row, 0]

    var h = seed_val ^ UInt32(b * 747796405 + 2891336453)
    h ^= h << 13
    h ^= h >> 17
    h ^= h << 5
    var dy = Int(h % 3) - 1
    h ^= h << 13
    h ^= h >> 17
    h ^= h << 5
    var dx = Int(h % 3) - 1

    for feat in range(tid, num_features, BLOCK_1D):
        var r, c = divmod(feat, 28)
        var sr = r - dy
        var sc = c - dx
        if sr >= 0 and sr < 28 and sc >= 0 and sc < 28:
            x_batch[b, feat] = rebind[x_batch.ElementType](
                x_all[src_row, sr * 28 + sc]
            )
        else:
            x_batch[b, feat] = x_batch.ElementType(0.0)


@inline(.always)
def gpu_gather_batch_rows_shifted[
    dtype: DType, total_samples: Int, num_features: Int, batch_size: Int
](
    ctx: DeviceContext,
    x_all: DeviceMatrix[dtype, total_samples, num_features],
    y_all: DeviceMatrix[DType.int32, total_samples, 1],
    indices: DeviceMatrix[DType.int32, total_samples, 1],
    batch_start: Int,
    seed_val: UInt32,
    mut x_batch: DeviceMatrix[dtype, batch_size, num_features],
    mut y_batch: DeviceMatrix[DType.int32, batch_size, 1],
) raises:
    comptime kernel = gather_batch_rows_shifted_kernel[
        dtype, total_samples, num_features, batch_size
    ]
    var tx = TileTensor(x_all.buffer, x_all.layout)
    var ty = TileTensor(y_all.buffer, y_all.layout)
    var tidx = TileTensor(indices.buffer, indices.layout)
    var txb = TileTensor(x_batch.buffer, x_batch.layout)
    var tyb = TileTensor(y_batch.buffer, y_batch.layout)
    ctx.enqueue_function[kernel](
        tx.as_imm(),
        ty.as_imm(),
        tidx.as_imm(),
        Int32(batch_start),
        seed_val,
        txb,
        tyb,
        grid_dim=batch_size,
        block_dim=BLOCK_1D,
    )


def dropout_inplace_kernel[
    dtype: DType, rows: Int, cols: Int
](
    a: MutTileTensor[
        dtype, type_of(row_major[rows, cols]()), MutUntrackedOrigin
    ],
    seed_val: UInt32,
    thresh: UInt32,
    inv_keep: Scalar[dtype],
):
    """Inverted dropout in-place using a stateless per-thread PRNG hash."""
    comptime assert a.flat_rank == 2
    comptime total = rows * cols
    var tid = global_idx.x
    if tid >= total:
        return
    var r, c = divmod(tid, cols)
    var h = seed_val ^ UInt32(tid * 747796405 + 2891336453)
    h ^= h << 13
    h ^= h >> 17
    h ^= h << 5
    if h < thresh:
        a[r, c] = rebind[a.ElementType](
            rebind[Scalar[dtype]](a[r, c]) * inv_keep
        )
    else:
        a[r, c] = a.ElementType(0.0)


@inline(.always)
def gpu_apply_dropout[
    dtype: DType, rows: Int, cols: Int
](
    ctx: DeviceContext,
    mut a: DeviceMatrix[dtype, rows, cols],
    seed_val: UInt32,
    keep_prob: Scalar[dtype],
    inv_keep: Scalar[dtype],
) raises:
    comptime total = rows * cols
    comptime kernel = dropout_inplace_kernel[dtype, rows, cols]
    var thresh = UInt32(Float64(keep_prob) * 4294967295.0)
    var ta = TileTensor(a.buffer, a.layout)
    ctx.enqueue_function[kernel](
        ta,
        seed_val,
        thresh,
        inv_keep,
        grid_dim=ceildiv(total, BLOCK_1D),
        block_dim=BLOCK_1D,
    )


def relu_mask_scale_inplace_kernel[
    dtype: DType, rows: Int, cols: Int
](
    dz: MutTileTensor[
        dtype, type_of(row_major[rows, cols]()), MutUntrackedOrigin
    ],
    mask_act: ImmTileTensor[
        dtype, type_of(row_major[rows, cols]()), ImmUntrackedOrigin
    ],
    scale: Scalar[dtype],
):
    """Computes dz = (mask_act > 0) ? dz * scale : 0.0."""
    comptime assert dz.flat_rank == 2 and mask_act.flat_rank == 2
    comptime total = rows * cols
    var tid = global_idx.x
    if tid >= total:
        return
    var r, c = divmod(tid, cols)
    if rebind[Scalar[dtype]](mask_act[r, c]) > 0.0:
        dz[r, c] = rebind[dz.ElementType](
            rebind[Scalar[dtype]](dz[r, c]) * scale
        )
    else:
        dz[r, c] = dz.ElementType(0.0)


@inline(.always)
def gpu_apply_relu_mask_scale[
    dtype: DType, rows: Int, cols: Int
](
    ctx: DeviceContext,
    mut dz: DeviceMatrix[dtype, rows, cols],
    mask_act: DeviceMatrix[dtype, rows, cols],
    scale: Scalar[dtype],
) raises:
    comptime total = rows * cols
    comptime kernel = relu_mask_scale_inplace_kernel[dtype, rows, cols]
    var tdz = TileTensor(dz.buffer, dz.layout)
    var tm = TileTensor(mask_act.buffer, mask_act.layout)
    ctx.enqueue_function[kernel](
        tdz,
        tm.as_imm(),
        scale,
        grid_dim=ceildiv(total, BLOCK_1D),
        block_dim=BLOCK_1D,
    )


def softmax_ce_grad_metrics_smooth_db_rows_kernel[
    dtype: DType, batch_size: Int, num_classes: Int, block_threads: Int
](
    z2: ImmTileTensor[
        dtype, type_of(row_major[batch_size, num_classes]()), ImmUntrackedOrigin
    ],
    labels: ImmTileTensor[
        DType.int32, type_of(row_major[batch_size, 1]()), ImmUntrackedOrigin
    ],
    dz2: MutTileTensor[
        dtype, type_of(row_major[batch_size, num_classes]()), MutUntrackedOrigin
    ],
    db2: MutTileTensor[
        dtype, type_of(row_major[num_classes, 1]()), MutUntrackedOrigin
    ],
    epoch_loss: MutTileTensor[
        dtype, type_of(row_major[1, 1]()), MutUntrackedOrigin
    ],
    epoch_correct: MutTileTensor[
        DType.int32, type_of(row_major[1, 1]()), MutUntrackedOrigin
    ],
    smoothing: Scalar[dtype],
):
    """Fused softmax + label-smoothed CE loss, scaled dZ/dB, and accuracy."""
    comptime assert dtype.is_floating_point()
    comptime assert (
        z2.flat_rank == 2
        and labels.flat_rank == 2
        and dz2.flat_rank == 2
        and db2.flat_rank == 2
        and epoch_loss.flat_rank == 2
        and epoch_correct.flat_rank == 2
    )
    comptime num_warps = block_threads // WARP_SIZE
    comptime inv_batch = Scalar[dtype](1.0) / Scalar[dtype](batch_size)

    var off_val = smoothing / Scalar[dtype](num_classes)
    var on_val = Scalar[dtype](1.0) - smoothing + off_val

    var tid = thread_idx.x
    var my_loss: Scalar[dtype] = 0.0
    var my_correct: Int32 = 0
    var grads = Array[Scalar[dtype], num_classes](fill=0.0)

    if tid < batch_size:
        var lbl = Int(labels[tid, 0])
        grads[0] = rebind[Scalar[dtype]](z2[tid, 0])
        var row_max = grads[0]
        var best_c = 0
        comptime for c in range(1, num_classes):
            var v = rebind[Scalar[dtype]](z2[tid, c])
            grads[c] = v
            if v > row_max:
                row_max = v
                best_c = c

        var sum_exp: Scalar[dtype] = 0.0
        comptime for c in range(num_classes):
            var e_val = exp(grads[c] - row_max)
            grads[c] = e_val
            sum_exp += e_val

        var inv_sum = Scalar[dtype](1.0) / sum_exp
        comptime for c in range(num_classes):
            var p = grads[c] * inv_sum
            var target = on_val if c == lbl else off_val
            my_loss -= target * log(max(p, Scalar[dtype](1e-7)))
            var g = (p - target) * inv_batch
            grads[c] = g
            dz2[tid, c] = rebind[dz2.ElementType](g)

        if best_c == lbl:
            my_correct = 1

    var warp_loss = warp.sum(my_loss)
    var warp_correct = warp.sum(my_correct)

    comptime if num_warps <= 1:
        comptime for c in range(num_classes):
            var w_db = warp.sum(grads[c])
            if tid == 0:
                db2[c, 0] = rebind[db2.ElementType](w_db)
        if tid == 0:
            epoch_loss[0, 0] = epoch_loss[0, 0] + rebind[
                epoch_loss.ElementType
            ](warp_loss * inv_batch)
            epoch_correct[0, 0] = epoch_correct[0, 0] + warp_correct
    else:
        var s_loss = stack_allocation[dtype, address_space=.SHARED](
            row_major[num_warps]()
        )
        var s_correct = stack_allocation[DType.int32, address_space=.SHARED](
            row_major[num_warps]()
        )
        var s_db2 = stack_allocation[dtype, address_space=.SHARED](
            row_major[num_warps, num_classes]()
        )
        var wid = tid // WARP_SIZE
        var lane = lane_id()
        if lane == 0:
            s_loss[wid] = rebind[s_loss.ElementType](warp_loss)
            s_correct[wid] = warp_correct
        comptime for c in range(num_classes):
            var w_db = warp.sum(grads[c])
            if lane == 0:
                s_db2[wid, c] = rebind[s_db2.ElementType](w_db)
        barrier()

        if tid == 0:
            var total_loss: Scalar[dtype] = 0.0
            var total_correct: Int32 = 0
            comptime for w in range(num_warps):
                total_loss += rebind[Scalar[dtype]](s_loss[w])
                total_correct += rebind[Int32](s_correct[w])
            epoch_loss[0, 0] = epoch_loss[0, 0] + rebind[
                epoch_loss.ElementType
            ](total_loss * inv_batch)
            epoch_correct[0, 0] = epoch_correct[0, 0] + total_correct

        if tid < num_classes:
            var total_db: Scalar[dtype] = 0.0
            comptime for w in range(num_warps):
                total_db += rebind[Scalar[dtype]](s_db2[w, tid])
            db2[tid, 0] = rebind[db2.ElementType](total_db)


@inline(.always)
def gpu_softmax_ce_grad_metrics_smooth_db_rows[
    dtype: DType, batch_size: Int, num_classes: Int
](
    ctx: DeviceContext,
    z2: DeviceMatrix[dtype, batch_size, num_classes],
    labels: DeviceMatrix[DType.int32, batch_size, 1],
    mut dz2: DeviceMatrix[dtype, batch_size, num_classes],
    mut db2: DeviceMatrix[dtype, num_classes, 1],
    mut epoch_loss: DeviceMatrix[dtype, 1, 1],
    mut epoch_correct: DeviceMatrix[DType.int32, 1, 1],
    smoothing: Scalar[dtype],
) raises:
    comptime block_threads = min(
        BLOCK_1D, align_up(batch_size, HOST_WARP_ALIGN)
    )
    comptime kernel = softmax_ce_grad_metrics_smooth_db_rows_kernel[
        dtype, batch_size, num_classes, block_threads
    ]
    var tz = TileTensor(z2.buffer, z2.layout)
    var tl = TileTensor(labels.buffer, labels.layout)
    var td = TileTensor(dz2.buffer, dz2.layout)
    var tdb = TileTensor(db2.buffer, db2.layout)
    var tel = TileTensor(epoch_loss.buffer, epoch_loss.layout)
    var tec = TileTensor(epoch_correct.buffer, epoch_correct.layout)
    ctx.enqueue_function[kernel](
        tz.as_imm(),
        tl.as_imm(),
        td,
        tdb,
        tel,
        tec,
        smoothing,
        grid_dim=1,
        block_dim=block_threads,
    )


def adamw_mlp3_kernel[
    dtype: DType, I: Int, H1: Int, H2: Int, C: Int
](
    w1: MutTileTensor[dtype, type_of(row_major[I, H1]()), MutUntrackedOrigin],
    dw1: ImmTileTensor[dtype, type_of(row_major[I, H1]()), ImmUntrackedOrigin],
    mw1: MutTileTensor[dtype, type_of(row_major[I, H1]()), MutUntrackedOrigin],
    vw1: MutTileTensor[dtype, type_of(row_major[I, H1]()), MutUntrackedOrigin],
    b1: MutTileTensor[dtype, type_of(row_major[H1, 1]()), MutUntrackedOrigin],
    db1: ImmTileTensor[dtype, type_of(row_major[H1, 1]()), ImmUntrackedOrigin],
    mb1: MutTileTensor[dtype, type_of(row_major[H1, 1]()), MutUntrackedOrigin],
    vb1: MutTileTensor[dtype, type_of(row_major[H1, 1]()), MutUntrackedOrigin],
    w2: MutTileTensor[dtype, type_of(row_major[H1, H2]()), MutUntrackedOrigin],
    dw2: ImmTileTensor[dtype, type_of(row_major[H1, H2]()), ImmUntrackedOrigin],
    mw2: MutTileTensor[dtype, type_of(row_major[H1, H2]()), MutUntrackedOrigin],
    vw2: MutTileTensor[dtype, type_of(row_major[H1, H2]()), MutUntrackedOrigin],
    b2: MutTileTensor[dtype, type_of(row_major[H2, 1]()), MutUntrackedOrigin],
    db2: ImmTileTensor[dtype, type_of(row_major[H2, 1]()), ImmUntrackedOrigin],
    mb2: MutTileTensor[dtype, type_of(row_major[H2, 1]()), MutUntrackedOrigin],
    vb2: MutTileTensor[dtype, type_of(row_major[H2, 1]()), MutUntrackedOrigin],
    w3: MutTileTensor[dtype, type_of(row_major[C, H2]()), MutUntrackedOrigin],
    dw3: ImmTileTensor[dtype, type_of(row_major[C, H2]()), ImmUntrackedOrigin],
    mw3: MutTileTensor[dtype, type_of(row_major[C, H2]()), MutUntrackedOrigin],
    vw3: MutTileTensor[dtype, type_of(row_major[C, H2]()), MutUntrackedOrigin],
    b3: MutTileTensor[dtype, type_of(row_major[C, 1]()), MutUntrackedOrigin],
    db3: ImmTileTensor[dtype, type_of(row_major[C, 1]()), ImmUntrackedOrigin],
    mb3: MutTileTensor[dtype, type_of(row_major[C, 1]()), MutUntrackedOrigin],
    vb3: MutTileTensor[dtype, type_of(row_major[C, 1]()), MutUntrackedOrigin],
    lr: Scalar[dtype],
    inv_bc1: Scalar[dtype],
    inv_bc2: Scalar[dtype],
    weight_decay: Scalar[dtype],
):
    """Fused AdamW update for all 6 3-layer MLP parameter tensors in 1 launch.
    """
    comptime assert (
        w1.flat_rank == 2
        and b1.flat_rank == 2
        and w2.flat_rank == 2
        and b2.flat_rank == 2
        and w3.flat_rank == 2
        and b3.flat_rank == 2
    )
    comptime n_w1 = I * H1
    comptime n_b1 = H1
    comptime n_w2 = H1 * H2
    comptime n_b2 = H2
    comptime n_w3 = C * H2
    comptime n_b3 = C
    comptime off_b1 = n_w1
    comptime off_w2 = off_b1 + n_b1
    comptime off_b2 = off_w2 + n_w2
    comptime off_w3 = off_b2 + n_b2
    comptime off_b3 = off_w3 + n_w3
    comptime total = off_b3 + n_b3

    comptime beta1 = Scalar[dtype](0.9)
    comptime beta2 = Scalar[dtype](0.999)
    comptime one_m_b1 = Scalar[dtype](0.1)
    comptime one_m_b2 = Scalar[dtype](0.001)
    comptime eps = Scalar[dtype](1e-8)

    var tid = global_idx.x
    if tid >= total:
        return

    var decay = Scalar[dtype](1.0) - lr * weight_decay

    @inline(.always)
    def step_val(
        p: Scalar[dtype],
        g: Scalar[dtype],
        m_old: Scalar[dtype],
        v_old: Scalar[dtype],
        use_decay: Bool,
    ) {imm lr, imm inv_bc1, imm inv_bc2, imm decay} -> Tuple[
        Scalar[dtype], Scalar[dtype], Scalar[dtype]
    ]:
        var m = m_old * beta1 + g * one_m_b1
        var v = v_old * beta2 + g * g * one_m_b2
        var m_hat = m * inv_bc1
        var v_hat = v * inv_bc2
        var p_base = p * decay if use_decay else p
        var p_new = p_base - lr * m_hat / (sqrt(v_hat) + eps)
        return p_new, m, v

    if tid < off_b1:
        var r, c = divmod(tid, H1)
        var pn, mn, vn = step_val(
            rebind[Scalar[dtype]](w1[r, c]),
            rebind[Scalar[dtype]](dw1[r, c]),
            rebind[Scalar[dtype]](mw1[r, c]),
            rebind[Scalar[dtype]](vw1[r, c]),
            True,
        )
        w1[r, c] = rebind[w1.ElementType](pn)
        mw1[r, c] = rebind[mw1.ElementType](mn)
        vw1[r, c] = rebind[vw1.ElementType](vn)
    elif tid < off_w2:
        var r = tid - off_b1
        var pn, mn, vn = step_val(
            rebind[Scalar[dtype]](b1[r, 0]),
            rebind[Scalar[dtype]](db1[r, 0]),
            rebind[Scalar[dtype]](mb1[r, 0]),
            rebind[Scalar[dtype]](vb1[r, 0]),
            False,
        )
        b1[r, 0] = rebind[b1.ElementType](pn)
        mb1[r, 0] = rebind[mb1.ElementType](mn)
        vb1[r, 0] = rebind[vb1.ElementType](vn)
    elif tid < off_b2:
        var r, c = divmod(tid - off_w2, H2)
        var pn, mn, vn = step_val(
            rebind[Scalar[dtype]](w2[r, c]),
            rebind[Scalar[dtype]](dw2[r, c]),
            rebind[Scalar[dtype]](mw2[r, c]),
            rebind[Scalar[dtype]](vw2[r, c]),
            True,
        )
        w2[r, c] = rebind[w2.ElementType](pn)
        mw2[r, c] = rebind[mw2.ElementType](mn)
        vw2[r, c] = rebind[vw2.ElementType](vn)
    elif tid < off_w3:
        var r = tid - off_b2
        var pn, mn, vn = step_val(
            rebind[Scalar[dtype]](b2[r, 0]),
            rebind[Scalar[dtype]](db2[r, 0]),
            rebind[Scalar[dtype]](mb2[r, 0]),
            rebind[Scalar[dtype]](vb2[r, 0]),
            False,
        )
        b2[r, 0] = rebind[b2.ElementType](pn)
        mb2[r, 0] = rebind[mb2.ElementType](mn)
        vb2[r, 0] = rebind[vb2.ElementType](vn)
    elif tid < off_b3:
        var r, c = divmod(tid - off_w3, H2)
        var pn, mn, vn = step_val(
            rebind[Scalar[dtype]](w3[r, c]),
            rebind[Scalar[dtype]](dw3[r, c]),
            rebind[Scalar[dtype]](mw3[r, c]),
            rebind[Scalar[dtype]](vw3[r, c]),
            True,
        )
        w3[r, c] = rebind[w3.ElementType](pn)
        mw3[r, c] = rebind[mw3.ElementType](mn)
        vw3[r, c] = rebind[vw3.ElementType](vn)
    else:
        var r = tid - off_b3
        var pn, mn, vn = step_val(
            rebind[Scalar[dtype]](b3[r, 0]),
            rebind[Scalar[dtype]](db3[r, 0]),
            rebind[Scalar[dtype]](mb3[r, 0]),
            rebind[Scalar[dtype]](vb3[r, 0]),
            False,
        )
        b3[r, 0] = rebind[b3.ElementType](pn)
        mb3[r, 0] = rebind[mb3.ElementType](mn)
        vb3[r, 0] = rebind[vb3.ElementType](vn)


@inline(.always)
def gpu_adamw_update_mlp3[
    dtype: DType, I: Int, H1: Int, H2: Int, C: Int
](
    ctx: DeviceContext,
    mut w1: DeviceMatrix[dtype, I, H1],
    dw1: DeviceMatrix[dtype, I, H1],
    mut mw1: DeviceMatrix[dtype, I, H1],
    mut vw1: DeviceMatrix[dtype, I, H1],
    mut b1: DeviceMatrix[dtype, H1, 1],
    db1: DeviceMatrix[dtype, H1, 1],
    mut mb1: DeviceMatrix[dtype, H1, 1],
    mut vb1: DeviceMatrix[dtype, H1, 1],
    mut w2: DeviceMatrix[dtype, H1, H2],
    dw2: DeviceMatrix[dtype, H1, H2],
    mut mw2: DeviceMatrix[dtype, H1, H2],
    mut vw2: DeviceMatrix[dtype, H1, H2],
    mut b2: DeviceMatrix[dtype, H2, 1],
    db2: DeviceMatrix[dtype, H2, 1],
    mut mb2: DeviceMatrix[dtype, H2, 1],
    mut vb2: DeviceMatrix[dtype, H2, 1],
    mut w3: DeviceMatrix[dtype, C, H2],
    dw3: DeviceMatrix[dtype, C, H2],
    mut mw3: DeviceMatrix[dtype, C, H2],
    mut vw3: DeviceMatrix[dtype, C, H2],
    mut b3: DeviceMatrix[dtype, C, 1],
    db3: DeviceMatrix[dtype, C, 1],
    mut mb3: DeviceMatrix[dtype, C, 1],
    mut vb3: DeviceMatrix[dtype, C, 1],
    lr: Scalar[dtype],
    inv_bc1: Scalar[dtype],
    inv_bc2: Scalar[dtype],
    weight_decay: Scalar[dtype],
) raises:
    comptime total = I * H1 + H1 + H1 * H2 + H2 + C * H2 + C
    comptime kernel = adamw_mlp3_kernel[dtype, I, H1, H2, C]
    ctx.enqueue_function[kernel](
        TileTensor(w1.buffer, w1.layout),
        TileTensor(dw1.buffer, dw1.layout).as_imm(),
        TileTensor(mw1.buffer, mw1.layout),
        TileTensor(vw1.buffer, vw1.layout),
        TileTensor(b1.buffer, b1.layout),
        TileTensor(db1.buffer, db1.layout).as_imm(),
        TileTensor(mb1.buffer, mb1.layout),
        TileTensor(vb1.buffer, vb1.layout),
        TileTensor(w2.buffer, w2.layout),
        TileTensor(dw2.buffer, dw2.layout).as_imm(),
        TileTensor(mw2.buffer, mw2.layout),
        TileTensor(vw2.buffer, vw2.layout),
        TileTensor(b2.buffer, b2.layout),
        TileTensor(db2.buffer, db2.layout).as_imm(),
        TileTensor(mb2.buffer, mb2.layout),
        TileTensor(vb2.buffer, vb2.layout),
        TileTensor(w3.buffer, w3.layout),
        TileTensor(dw3.buffer, dw3.layout).as_imm(),
        TileTensor(mw3.buffer, mw3.layout),
        TileTensor(vw3.buffer, vw3.layout),
        TileTensor(b3.buffer, b3.layout),
        TileTensor(db3.buffer, db3.layout).as_imm(),
        TileTensor(mb3.buffer, mb3.layout),
        TileTensor(vb3.buffer, vb3.layout),
        lr,
        inv_bc1,
        inv_bc2,
        weight_decay,
        grid_dim=ceildiv(total, BLOCK_1D),
        block_dim=BLOCK_1D,
    )
