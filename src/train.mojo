from std.math import cos, pi
from std.memory import unsafe_memcpy
from std.pathlib import Path
from std.random import seed, shuffle
from std.sys import get_defined_bool, has_accelerator
from std.sys.info import CompilationTarget
from std.time import perf_counter
from max.gpu.host import DeviceContext

from load_file import MNISTDataset, load_mnist_csv
from weights import save_weights_mlp3
from cpu_ops import (
    CPUMatrix,
    StaticMatrixView,
    cpu_adamw_update_static,
    cpu_apply_dropout_static,
    cpu_apply_relu_mask_scale_static,
    cpu_eval_batch_logits_rows,
    cpu_gather_batch_rows_shifted,
    cpu_matmul_nn_bias_row_relu_static,
    cpu_matmul_nn_static,
    cpu_matmul_nt_bias_row_static,
    cpu_matmul_nt_static,
    cpu_matmul_tn_static,
    cpu_softmax_ce_grad_metrics_smooth_rows,
    cpu_sum_axis0_scaled_static,
)
from gpu_mem import DeviceMatrix
from gpu_ops import (
    gpu_adamw_update_mlp3,
    gpu_apply_dropout,
    gpu_apply_relu_mask_scale,
    gpu_eval_argmax_rows_count,
    gpu_gather_batch_rows_shifted,
    gpu_matmul_nn,
    gpu_matmul_nn_bias_row_relu,
    gpu_matmul_nn_offset_bias_row_relu,
    gpu_matmul_nt,
    gpu_matmul_nt_bias_row,
    gpu_matmul_tn,
    gpu_softmax_ce_grad_metrics_smooth_db_rows,
    gpu_sum_axis0_scaled,
)


comptime IMG_ROWS = 28
comptime IMG_COLS = 28
comptime IMG_PIXELS = IMG_ROWS * IMG_COLS
comptime NUM_CLASSES = 10
comptime HIDDEN1 = 512
comptime HIDDEN2 = 256

comptime TRAIN_FILEPATH = "digit-recognizer/train.csv"
comptime TEST_FILEPATH = "digit-recognizer/test.csv"
comptime PREDICTIONS_FILEPATH = "predictions.csv"
comptime WEIGHTS_FILEPATH = "mnist_weights.bin"

comptime TOTAL_TRAIN_ROWS = 42000
comptime TRAIN_SPLIT = 40000
comptime VAL_SPLIT = TOTAL_TRAIN_ROWS - TRAIN_SPLIT  # 2000
comptime TEST_ROWS = 28000

comptime BATCH_SIZE = 200
comptime CPU_EVAL_BATCH = 500
comptime GPU_EVAL_BATCH = 2000
comptime EPOCHS = 25

comptime LR_MAX = Float64(2e-3)
comptime LR_MIN = Float64(1e-4)
comptime BETA1 = Float64(0.9)
comptime BETA2 = Float64(0.999)
comptime WEIGHT_DECAY = Float32(1e-4)
comptime DROPOUT_KEEP = Float32(0.8)
comptime DROPOUT_INV_KEEP = Float32(1.0 / 0.8)
comptime LABEL_SMOOTHING = Float32(0.05)

comptime TRAIN_NUM_BATCHES = TRAIN_SPLIT // BATCH_SIZE
comptime TOTAL_STEPS = EPOCHS * TRAIN_NUM_BATCHES
comptime INV_NUM_BATCHES = Float32(1.0) / Float32(TRAIN_NUM_BATCHES)
comptime INV_TRAIN_SPLIT_PCT = Float32(100.0) / Float32(TRAIN_SPLIT)
comptime INV_VAL_SPLIT_PCT = Float32(100.0) / Float32(VAL_SPLIT)

comptime USE_GPU = has_accelerator() and get_defined_bool[
    "USE_GPU", not CompilationTarget.is_macos()
]()


@inline(.always)
def cosine_lr(step_num: Int) -> Float32:
    var progress = Float64(step_num - 1) / Float64(max(1, TOTAL_STEPS - 1))
    return Float32(
        LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + cos(pi * progress))
    )


def write_submission_csv(path: String, preds: Span[Int, _]) raises:
    """Writes Kaggle Digit Recognizer submission CSV: ImageId,Label."""
    var out = String("ImageId,Label\n")
    for i in range(len(preds)):
        out.write(i + 1, ",", preds[i], "\n")
    Path(path).write_text(out)


def print_sample_predictions(
    dataset: MNISTDataset,
    preds: Span[Int, _],
    start_idx: Int,
    count: Int,
    title: String,
):
    print("\n===", title, "===")
    for i in range(count):
        var idx = start_idx + i
        if dataset.has_labels:
            print(
                t"Sample #{idx}: predicted={preds[idx]},"
                t" actual={dataset.labels[idx]}"
            )
        else:
            print(t"Test ImageId #{idx + 1}: predicted={preds[idx]}")
        print(dataset.ascii_preview(idx, rows=IMG_ROWS, cols=IMG_COLS))


@inline(.always)
def _copy_cpu_matrix(src: CPUMatrix, mut dst: CPUMatrix):
    unsafe_memcpy(
        dest=dst.data.unsafe_ptr(),
        src=src.data.unsafe_ptr(),
        count=len(src.data),
    )


def evaluate_cpu[
    has_labels: Bool
](
    dataset: MNISTDataset,
    start_idx: Int,
    num_samples: Int,
    w1: CPUMatrix,
    b1: CPUMatrix,
    w2: CPUMatrix,
    b2: CPUMatrix,
    w3: CPUMatrix,
    b3: CPUMatrix,
    mut a1_eval: CPUMatrix,
    mut a2_eval: CPUMatrix,
    mut z3_eval: CPUMatrix,
    mut out_preds: List[Int],
) raises -> Int:
    """Zero-copy sample-major 3-layer MLP forward inference on CPU."""
    var w1_v = w1.static_view[IMG_PIXELS, HIDDEN1]()
    var b1_v = b1.static_view[HIDDEN1, 1]()
    var w2_v = w2.static_view[HIDDEN1, HIDDEN2]()
    var b2_v = b2.static_view[HIDDEN2, 1]()
    var w3_v = w3.static_view[NUM_CLASSES, HIDDEN2]()
    var b3_v = b3.static_view[NUM_CLASSES, 1]()
    var a1_mv = a1_eval.static_mut_view[CPU_EVAL_BATCH, HIDDEN1]()
    var a1_v = a1_eval.static_view[CPU_EVAL_BATCH, HIDDEN1]()
    var a2_mv = a2_eval.static_mut_view[CPU_EVAL_BATCH, HIDDEN2]()
    var a2_v = a2_eval.static_view[CPU_EVAL_BATCH, HIDDEN2]()
    var z3_mv = z3_eval.static_mut_view[CPU_EVAL_BATCH, NUM_CLASSES]()
    var z3_v = z3_eval.static_view[CPU_EVAL_BATCH, NUM_CLASSES]()

    var total_correct = 0
    var pixels_ptr = Span(dataset.pixels).unsafe_ptr()
    var labels_span = Span(dataset.labels)
    var num_batches = num_samples // CPU_EVAL_BATCH

    for batch_idx in range(num_batches):
        var offset = start_idx + batch_idx * CPU_EVAL_BATCH
        var x_v = StaticMatrixView[CPU_EVAL_BATCH, IMG_PIXELS](
            pixels_ptr.unsafe_offset(offset * IMG_PIXELS)
        )
        cpu_matmul_nn_bias_row_relu_static[CPU_EVAL_BATCH, HIDDEN1, IMG_PIXELS](
            x_v, w1_v, a1_mv, b1_v
        )
        cpu_matmul_nn_bias_row_relu_static[CPU_EVAL_BATCH, HIDDEN2, HIDDEN1](
            a1_v, w2_v, a2_mv, b2_v
        )
        cpu_matmul_nt_bias_row_static[CPU_EVAL_BATCH, NUM_CLASSES, HIDDEN2](
            a2_v, w3_v, z3_mv, b3_v
        )
        total_correct += cpu_eval_batch_logits_rows[
            CPU_EVAL_BATCH, NUM_CLASSES, has_labels
        ](z3_v, out_preds, offset, labels_span)

    return total_correct


def train_and_predict_cpu(train_ds: MNISTDataset, test_ds: MNISTDataset) raises:
    print("Running on CPU (multi-threaded SIMD)")
    comptime if CompilationTarget.is_macos() and has_accelerator():
        print(
            "Note: On macOS, CPU runs by default. To run on Apple GPU,"
            " run `TOOLCHAINS=... pixi run train-gpu`."
        )

    var w1 = CPUMatrix(IMG_PIXELS, HIDDEN1)
    var b1 = CPUMatrix(HIDDEN1, 1, fill=Float32(0.0))
    var w2 = CPUMatrix(HIDDEN1, HIDDEN2)
    var b2 = CPUMatrix(HIDDEN2, 1, fill=Float32(0.0))
    var w3 = CPUMatrix(NUM_CLASSES, HIDDEN2)
    var b3 = CPUMatrix(NUM_CLASSES, 1, fill=Float32(0.0))

    w1.init_he(IMG_PIXELS)
    w2.init_he(HIDDEN1)
    w3.init_he(HIDDEN2)

    var best_w1 = CPUMatrix(IMG_PIXELS, HIDDEN1)
    var best_b1 = CPUMatrix(HIDDEN1, 1)
    var best_w2 = CPUMatrix(HIDDEN1, HIDDEN2)
    var best_b2 = CPUMatrix(HIDDEN2, 1)
    var best_w3 = CPUMatrix(NUM_CLASSES, HIDDEN2)
    var best_b3 = CPUMatrix(NUM_CLASSES, 1)

    # AdamW first and second moments
    var mw1 = CPUMatrix(IMG_PIXELS, HIDDEN1)
    var vw1 = CPUMatrix(IMG_PIXELS, HIDDEN1)
    var mb1 = CPUMatrix(HIDDEN1, 1)
    var vb1 = CPUMatrix(HIDDEN1, 1)
    var mw2 = CPUMatrix(HIDDEN1, HIDDEN2)
    var vw2 = CPUMatrix(HIDDEN1, HIDDEN2)
    var mb2 = CPUMatrix(HIDDEN2, 1)
    var vb2 = CPUMatrix(HIDDEN2, 1)
    var mw3 = CPUMatrix(NUM_CLASSES, HIDDEN2)
    var vw3 = CPUMatrix(NUM_CLASSES, HIDDEN2)
    var mb3 = CPUMatrix(NUM_CLASSES, 1)
    var vb3 = CPUMatrix(NUM_CLASSES, 1)

    var dw1 = CPUMatrix(IMG_PIXELS, HIDDEN1)
    var db1 = CPUMatrix(HIDDEN1, 1)
    var dw2 = CPUMatrix(HIDDEN1, HIDDEN2)
    var db2 = CPUMatrix(HIDDEN2, 1)
    var dw3 = CPUMatrix(NUM_CLASSES, HIDDEN2)
    var db3 = CPUMatrix(NUM_CLASSES, 1)

    var x_b = CPUMatrix(BATCH_SIZE, IMG_PIXELS)
    var a1 = CPUMatrix(BATCH_SIZE, HIDDEN1)
    var a2 = CPUMatrix(BATCH_SIZE, HIDDEN2)
    var z3 = CPUMatrix(BATCH_SIZE, NUM_CLASSES)
    var dz3 = CPUMatrix(BATCH_SIZE, NUM_CLASSES)
    var dz2 = CPUMatrix(BATCH_SIZE, HIDDEN2)
    var dz1 = CPUMatrix(BATCH_SIZE, HIDDEN1)

    var a1_eval = CPUMatrix(CPU_EVAL_BATCH, HIDDEN1)
    var a2_eval = CPUMatrix(CPU_EVAL_BATCH, HIDDEN2)
    var z3_eval = CPUMatrix(CPU_EVAL_BATCH, NUM_CLASSES)

    var w1_v = w1.static_view[IMG_PIXELS, HIDDEN1]()
    var w1_mv = w1.static_mut_view[IMG_PIXELS, HIDDEN1]()
    var b1_v = b1.static_view[HIDDEN1, 1]()
    var b1_mv = b1.static_mut_view[HIDDEN1, 1]()
    var w2_v = w2.static_view[HIDDEN1, HIDDEN2]()
    var w2_mv = w2.static_mut_view[HIDDEN1, HIDDEN2]()
    var b2_v = b2.static_view[HIDDEN2, 1]()
    var b2_mv = b2.static_mut_view[HIDDEN2, 1]()
    var w3_v = w3.static_view[NUM_CLASSES, HIDDEN2]()
    var w3_mv = w3.static_mut_view[NUM_CLASSES, HIDDEN2]()
    var b3_v = b3.static_view[NUM_CLASSES, 1]()
    var b3_mv = b3.static_mut_view[NUM_CLASSES, 1]()

    var mw1_mv = mw1.static_mut_view[IMG_PIXELS, HIDDEN1]()
    var vw1_mv = vw1.static_mut_view[IMG_PIXELS, HIDDEN1]()
    var mb1_mv = mb1.static_mut_view[HIDDEN1, 1]()
    var vb1_mv = vb1.static_mut_view[HIDDEN1, 1]()
    var mw2_mv = mw2.static_mut_view[HIDDEN1, HIDDEN2]()
    var vw2_mv = vw2.static_mut_view[HIDDEN1, HIDDEN2]()
    var mb2_mv = mb2.static_mut_view[HIDDEN2, 1]()
    var vb2_mv = vb2.static_mut_view[HIDDEN2, 1]()
    var mw3_mv = mw3.static_mut_view[NUM_CLASSES, HIDDEN2]()
    var vw3_mv = vw3.static_mut_view[NUM_CLASSES, HIDDEN2]()
    var mb3_mv = mb3.static_mut_view[NUM_CLASSES, 1]()
    var vb3_mv = vb3.static_mut_view[NUM_CLASSES, 1]()

    var dw1_v = dw1.static_view[IMG_PIXELS, HIDDEN1]()
    var dw1_mv = dw1.static_mut_view[IMG_PIXELS, HIDDEN1]()
    var db1_v = db1.static_view[HIDDEN1, 1]()
    var db1_mv = db1.static_mut_view[HIDDEN1, 1]()
    var dw2_v = dw2.static_view[HIDDEN1, HIDDEN2]()
    var dw2_mv = dw2.static_mut_view[HIDDEN1, HIDDEN2]()
    var db2_v = db2.static_view[HIDDEN2, 1]()
    var db2_mv = db2.static_mut_view[HIDDEN2, 1]()
    var dw3_v = dw3.static_view[NUM_CLASSES, HIDDEN2]()
    var dw3_mv = dw3.static_mut_view[NUM_CLASSES, HIDDEN2]()
    var db3_v = db3.static_view[NUM_CLASSES, 1]()
    var db3_mv = db3.static_mut_view[NUM_CLASSES, 1]()

    var x_v = x_b.static_view[BATCH_SIZE, IMG_PIXELS]()
    var a1_v = a1.static_view[BATCH_SIZE, HIDDEN1]()
    var a1_mv = a1.static_mut_view[BATCH_SIZE, HIDDEN1]()
    var a2_v = a2.static_view[BATCH_SIZE, HIDDEN2]()
    var a2_mv = a2.static_mut_view[BATCH_SIZE, HIDDEN2]()
    var z3_v = z3.static_view[BATCH_SIZE, NUM_CLASSES]()
    var z3_mv = z3.static_mut_view[BATCH_SIZE, NUM_CLASSES]()
    var dz3_v = dz3.static_view[BATCH_SIZE, NUM_CLASSES]()
    var dz3_mv = dz3.static_mut_view[BATCH_SIZE, NUM_CLASSES]()
    var dz2_v = dz2.static_view[BATCH_SIZE, HIDDEN2]()
    var dz2_mv = dz2.static_mut_view[BATCH_SIZE, HIDDEN2]()
    var dz1_v = dz1.static_view[BATCH_SIZE, HIDDEN1]()
    var dz1_mv = dz1.static_mut_view[BATCH_SIZE, HIDDEN1]()

    var batch_labels = List[Int](length=BATCH_SIZE, fill=0)
    var indices = List[Int](capacity=TRAIN_SPLIT)
    for i in range(TRAIN_SPLIT):
        indices.append(i)

    var val_preds = List[Int](length=TOTAL_TRAIN_ROWS, fill=0)
    var pixels_span = Span(train_ds.pixels)
    var labels_span = Span(train_ds.labels)

    print(
        t"\nTraining 3-layer MLP ({IMG_PIXELS} -> {HIDDEN1} -> {HIDDEN2} ->"
        t" {NUM_CLASSES}) for {EPOCHS} epochs (batch_size={BATCH_SIZE},"
        t" AdamW cosine LR {LR_MAX}->{LR_MIN}, dropout=0.2,"
        t" label_smoothing={LABEL_SMOOTHING})..."
    )

    var rng_state = UInt32(0x9E3779B9)
    var b1_pow = Float64(1.0)
    var b2_pow = Float64(1.0)
    var step_num = 0
    var best_val_correct = -1

    var train_start = perf_counter()
    for epoch in range(1, EPOCHS + 1):
        var ep_start = perf_counter()
        shuffle(indices)
        var idx_span = Span(indices)

        var epoch_loss = Float32(0.0)
        var epoch_correct = 0

        for b_idx in range(TRAIN_NUM_BATCHES):
            step_num += 1
            b1_pow *= BETA1
            b2_pow *= BETA2
            var lr = cosine_lr(step_num)
            var inv_bc1 = Float32(1.0 / (1.0 - b1_pow))
            var inv_bc2 = Float32(1.0 / (1.0 - b2_pow))

            var batch_start = b_idx * BATCH_SIZE
            cpu_gather_batch_rows_shifted(
                pixels_span,
                idx_span,
                batch_start,
                BATCH_SIZE,
                x_b,
                batch_labels,
                labels_span,
                rng_state,
            )

            # Forward + inverted dropout
            cpu_matmul_nn_bias_row_relu_static[BATCH_SIZE, HIDDEN1, IMG_PIXELS](
                x_v, w1_v, a1_mv, b1_v
            )
            cpu_apply_dropout_static[BATCH_SIZE, HIDDEN1](
                a1_mv,
                UInt32(step_num * 2654435761),
                DROPOUT_KEEP,
                DROPOUT_INV_KEEP,
            )

            cpu_matmul_nn_bias_row_relu_static[BATCH_SIZE, HIDDEN2, HIDDEN1](
                a1_v, w2_v, a2_mv, b2_v
            )
            cpu_apply_dropout_static[BATCH_SIZE, HIDDEN2](
                a2_mv,
                UInt32(step_num * 2246822519 + 1),
                DROPOUT_KEEP,
                DROPOUT_INV_KEEP,
            )

            cpu_matmul_nt_bias_row_static[BATCH_SIZE, NUM_CLASSES, HIDDEN2](
                a2_v, w3_v, z3_mv, b3_v
            )

            # Softmax + label-smoothed CE loss, scaled dZ3/dB3, and accuracy
            var batch_loss, batch_correct = (
                cpu_softmax_ce_grad_metrics_smooth_rows[
                    BATCH_SIZE, NUM_CLASSES
                ](z3_v, Span(batch_labels), dz3_mv, db3_mv, LABEL_SMOOTHING)
            )
            epoch_loss += batch_loss
            epoch_correct += batch_correct

            # Backward pass (dZ3 and dB3 already scaled by 1/BATCH_SIZE)
            cpu_matmul_tn_static[NUM_CLASSES, HIDDEN2, BATCH_SIZE](
                dz3_v, a2_v, dw3_mv
            )

            cpu_matmul_nn_static[BATCH_SIZE, HIDDEN2, NUM_CLASSES](
                dz3_v, w3_v, dz2_mv
            )
            cpu_apply_relu_mask_scale_static[BATCH_SIZE, HIDDEN2](
                dz2_mv, a2_v, DROPOUT_INV_KEEP
            )
            cpu_sum_axis0_scaled_static[BATCH_SIZE, HIDDEN2](
                dz2_v, Float32(1.0), db2_mv
            )
            cpu_matmul_tn_static[HIDDEN1, HIDDEN2, BATCH_SIZE](
                a1_v, dz2_v, dw2_mv
            )

            cpu_matmul_nt_static[BATCH_SIZE, HIDDEN1, HIDDEN2](
                dz2_v, w2_v, dz1_mv
            )
            cpu_apply_relu_mask_scale_static[BATCH_SIZE, HIDDEN1](
                dz1_mv, a1_v, DROPOUT_INV_KEEP
            )
            cpu_sum_axis0_scaled_static[BATCH_SIZE, HIDDEN1](
                dz1_v, Float32(1.0), db1_mv
            )
            cpu_matmul_tn_static[IMG_PIXELS, HIDDEN1, BATCH_SIZE](
                x_v, dz1_v, dw1_mv
            )

            # AdamW parameter updates
            cpu_adamw_update_static[IMG_PIXELS, HIDDEN1](
                w1_mv, dw1_v, mw1_mv, vw1_mv, lr, inv_bc1, inv_bc2, WEIGHT_DECAY
            )
            cpu_adamw_update_static[HIDDEN1, 1](
                b1_mv, db1_v, mb1_mv, vb1_mv, lr, inv_bc1, inv_bc2, Float32(0.0)
            )
            cpu_adamw_update_static[HIDDEN1, HIDDEN2](
                w2_mv, dw2_v, mw2_mv, vw2_mv, lr, inv_bc1, inv_bc2, WEIGHT_DECAY
            )
            cpu_adamw_update_static[HIDDEN2, 1](
                b2_mv, db2_v, mb2_mv, vb2_mv, lr, inv_bc1, inv_bc2, Float32(0.0)
            )
            cpu_adamw_update_static[NUM_CLASSES, HIDDEN2](
                w3_mv, dw3_v, mw3_mv, vw3_mv, lr, inv_bc1, inv_bc2, WEIGHT_DECAY
            )
            cpu_adamw_update_static[NUM_CLASSES, 1](
                b3_mv, db3_v, mb3_mv, vb3_mv, lr, inv_bc1, inv_bc2, Float32(0.0)
            )

        var val_correct = evaluate_cpu[True](
            train_ds,
            TRAIN_SPLIT,
            VAL_SPLIT,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            a1_eval,
            a2_eval,
            z3_eval,
            val_preds,
        )
        if val_correct > best_val_correct:
            best_val_correct = val_correct
            _copy_cpu_matrix(w1, best_w1)
            _copy_cpu_matrix(b1, best_b1)
            _copy_cpu_matrix(w2, best_w2)
            _copy_cpu_matrix(b2, best_b2)
            _copy_cpu_matrix(w3, best_w3)
            _copy_cpu_matrix(b3, best_b3)

        var avg_loss = epoch_loss * INV_NUM_BATCHES
        var train_acc = Float32(epoch_correct) * INV_TRAIN_SPLIT_PCT
        var val_acc = Float32(val_correct) * INV_VAL_SPLIT_PCT
        var ep_time = perf_counter() - ep_start
        print(
            t"Epoch {epoch}/{EPOCHS} | loss={avg_loss} | train_acc={train_acc}%"
            t" | val_acc={val_acc}% | time={ep_time}s"
        )

    var total_train_time = perf_counter() - train_start
    print(t"\nTraining finished in {total_train_time}s!")

    var final_val_correct = evaluate_cpu[True](
        train_ds,
        TRAIN_SPLIT,
        VAL_SPLIT,
        best_w1,
        best_b1,
        best_w2,
        best_b2,
        best_w3,
        best_b3,
        a1_eval,
        a2_eval,
        z3_eval,
        val_preds,
    )
    var final_val_acc = Float32(final_val_correct) * INV_VAL_SPLIT_PCT
    print(
        t"Best Validation Accuracy: {final_val_correct}/{VAL_SPLIT}"
        t" ({final_val_acc}%)"
    )

    print_sample_predictions(
        train_ds,
        Span(val_preds),
        TRAIN_SPLIT,
        4,
        "Validation Sample Predictions",
    )

    var test_preds = List[Int](length=TEST_ROWS, fill=0)
    _ = evaluate_cpu[False](
        test_ds,
        0,
        TEST_ROWS,
        best_w1,
        best_b1,
        best_w2,
        best_b2,
        best_w3,
        best_b3,
        a1_eval,
        a2_eval,
        z3_eval,
        test_preds,
    )
    write_submission_csv(PREDICTIONS_FILEPATH, Span(test_preds))
    print(t"\nWrote {TEST_ROWS} test predictions to {PREDICTIONS_FILEPATH}")

    print_sample_predictions(
        test_ds,
        Span(test_preds),
        0,
        4,
        "Test Set Predictions (digit-recognizer/test.csv)",
    )

    save_weights_mlp3(
        WEIGHTS_FILEPATH, best_w1, best_b1, best_w2, best_b2, best_w3, best_b3
    )
    print(t"Saved trained weights to {WEIGHTS_FILEPATH}")


def evaluate_gpu[
    num_samples: Int, has_labels: Bool
](
    ctx: DeviceContext,
    x_all: DeviceMatrix[DType.float32, num_samples, IMG_PIXELS],
    y_all: DeviceMatrix[DType.int32, num_samples, 1],
    w1: DeviceMatrix[DType.float32, IMG_PIXELS, HIDDEN1],
    b1: DeviceMatrix[DType.float32, HIDDEN1, 1],
    w2: DeviceMatrix[DType.float32, HIDDEN1, HIDDEN2],
    b2: DeviceMatrix[DType.float32, HIDDEN2, 1],
    w3: DeviceMatrix[DType.float32, NUM_CLASSES, HIDDEN2],
    b3: DeviceMatrix[DType.float32, NUM_CLASSES, 1],
    mut a1_eval: DeviceMatrix[DType.float32, GPU_EVAL_BATCH, HIDDEN1],
    mut a2_eval: DeviceMatrix[DType.float32, GPU_EVAL_BATCH, HIDDEN2],
    mut z3_eval: DeviceMatrix[DType.float32, GPU_EVAL_BATCH, NUM_CLASSES],
    mut out_preds: DeviceMatrix[DType.int32, num_samples, 1],
    mut total_correct_dev: DeviceMatrix[DType.int32, 1, 1],
) raises -> Int:
    """Zero-copy sample-major 3-layer evaluation on GPU."""
    total_correct_dev.fill(0)
    comptime num_batches = num_samples // GPU_EVAL_BATCH

    for batch_idx in range(num_batches):
        var offset = batch_idx * GPU_EVAL_BATCH
        gpu_matmul_nn_offset_bias_row_relu(ctx, x_all, offset, w1, b1, a1_eval)
        gpu_matmul_nn_bias_row_relu(ctx, a1_eval, w2, b2, a2_eval)
        gpu_matmul_nt_bias_row(ctx, a2_eval, w3, b3, z3_eval)
        gpu_eval_argmax_rows_count[
            DType.float32, GPU_EVAL_BATCH, NUM_CLASSES, num_samples, has_labels
        ](ctx, z3_eval, y_all, offset, out_preds, total_correct_dev)

    comptime if has_labels:
        with total_correct_dev.buffer.map_to_host() as host_c:
            return Int(host_c[0])
    else:
        return 0


def train_and_predict_gpu(train_ds: MNISTDataset, test_ds: MNISTDataset) raises:
    var ctx = DeviceContext()
    print("Running on GPU:", ctx.device().get_name(), "api:", ctx.api())

    var x_train_dev = DeviceMatrix[DType.float32, TRAIN_SPLIT, IMG_PIXELS](ctx)
    var y_train_dev = DeviceMatrix[DType.int32, TRAIN_SPLIT, 1](ctx)
    var x_val_dev = DeviceMatrix[DType.float32, VAL_SPLIT, IMG_PIXELS](ctx)
    var y_val_dev = DeviceMatrix[DType.int32, VAL_SPLIT, 1](ctx)
    var x_test_dev = DeviceMatrix[DType.float32, TEST_ROWS, IMG_PIXELS](ctx)
    var y_test_dummy = DeviceMatrix[DType.int32, TEST_ROWS, 1](ctx, fill=0)

    var train_pixels_span = Span(train_ds.pixels)[0 : TRAIN_SPLIT * IMG_PIXELS]
    var val_pixels_span = Span(train_ds.pixels)[
        TRAIN_SPLIT * IMG_PIXELS : TOTAL_TRAIN_ROWS * IMG_PIXELS
    ]
    x_train_dev.copy_from_host(train_pixels_span)
    x_val_dev.copy_from_host(val_pixels_span)
    x_test_dev.copy_from_host(Span(test_ds.pixels))

    var y_train_i32 = List[Int32](capacity=TRAIN_SPLIT)
    for i in range(TRAIN_SPLIT):
        y_train_i32.append(Int32(train_ds.labels[i]))
    y_train_dev.copy_from_host(Span(y_train_i32))

    var y_val_i32 = List[Int32](capacity=VAL_SPLIT)
    for i in range(TRAIN_SPLIT, TOTAL_TRAIN_ROWS):
        y_val_i32.append(Int32(train_ds.labels[i]))
    y_val_dev.copy_from_host(Span(y_val_i32))

    var w1 = DeviceMatrix[DType.float32, IMG_PIXELS, HIDDEN1](ctx)
    var b1 = DeviceMatrix[DType.float32, HIDDEN1, 1](ctx, fill=0.0)
    var w2 = DeviceMatrix[DType.float32, HIDDEN1, HIDDEN2](ctx)
    var b2 = DeviceMatrix[DType.float32, HIDDEN2, 1](ctx, fill=0.0)
    var w3 = DeviceMatrix[DType.float32, NUM_CLASSES, HIDDEN2](ctx)
    var b3 = DeviceMatrix[DType.float32, NUM_CLASSES, 1](ctx, fill=0.0)

    w1.init_he(ctx, IMG_PIXELS)
    w2.init_he(ctx, HIDDEN1)
    w3.init_he(ctx, HIDDEN2)

    var best_hw1 = CPUMatrix(IMG_PIXELS, HIDDEN1)
    var best_hb1 = CPUMatrix(HIDDEN1, 1)
    var best_hw2 = CPUMatrix(HIDDEN1, HIDDEN2)
    var best_hb2 = CPUMatrix(HIDDEN2, 1)
    var best_hw3 = CPUMatrix(NUM_CLASSES, HIDDEN2)
    var best_hb3 = CPUMatrix(NUM_CLASSES, 1)

    var mw1 = DeviceMatrix[DType.float32, IMG_PIXELS, HIDDEN1](ctx, fill=0.0)
    var vw1 = DeviceMatrix[DType.float32, IMG_PIXELS, HIDDEN1](ctx, fill=0.0)
    var mb1 = DeviceMatrix[DType.float32, HIDDEN1, 1](ctx, fill=0.0)
    var vb1 = DeviceMatrix[DType.float32, HIDDEN1, 1](ctx, fill=0.0)
    var mw2 = DeviceMatrix[DType.float32, HIDDEN1, HIDDEN2](ctx, fill=0.0)
    var vw2 = DeviceMatrix[DType.float32, HIDDEN1, HIDDEN2](ctx, fill=0.0)
    var mb2 = DeviceMatrix[DType.float32, HIDDEN2, 1](ctx, fill=0.0)
    var vb2 = DeviceMatrix[DType.float32, HIDDEN2, 1](ctx, fill=0.0)
    var mw3 = DeviceMatrix[DType.float32, NUM_CLASSES, HIDDEN2](ctx, fill=0.0)
    var vw3 = DeviceMatrix[DType.float32, NUM_CLASSES, HIDDEN2](ctx, fill=0.0)
    var mb3 = DeviceMatrix[DType.float32, NUM_CLASSES, 1](ctx, fill=0.0)
    var vb3 = DeviceMatrix[DType.float32, NUM_CLASSES, 1](ctx, fill=0.0)

    var dw1 = DeviceMatrix[DType.float32, IMG_PIXELS, HIDDEN1](ctx)
    var db1 = DeviceMatrix[DType.float32, HIDDEN1, 1](ctx)
    var dw2 = DeviceMatrix[DType.float32, HIDDEN1, HIDDEN2](ctx)
    var db2 = DeviceMatrix[DType.float32, HIDDEN2, 1](ctx)
    var dw3 = DeviceMatrix[DType.float32, NUM_CLASSES, HIDDEN2](ctx)
    var db3 = DeviceMatrix[DType.float32, NUM_CLASSES, 1](ctx)

    var x_b = DeviceMatrix[DType.float32, BATCH_SIZE, IMG_PIXELS](ctx)
    var a1 = DeviceMatrix[DType.float32, BATCH_SIZE, HIDDEN1](ctx)
    var a2 = DeviceMatrix[DType.float32, BATCH_SIZE, HIDDEN2](ctx)
    var z3 = DeviceMatrix[DType.float32, BATCH_SIZE, NUM_CLASSES](ctx)
    var dz3 = DeviceMatrix[DType.float32, BATCH_SIZE, NUM_CLASSES](ctx)
    var dz2 = DeviceMatrix[DType.float32, BATCH_SIZE, HIDDEN2](ctx)
    var dz1 = DeviceMatrix[DType.float32, BATCH_SIZE, HIDDEN1](ctx)
    var labels_dev = DeviceMatrix[DType.int32, BATCH_SIZE, 1](ctx)

    var a1_eval = DeviceMatrix[DType.float32, GPU_EVAL_BATCH, HIDDEN1](ctx)
    var a2_eval = DeviceMatrix[DType.float32, GPU_EVAL_BATCH, HIDDEN2](ctx)
    var z3_eval = DeviceMatrix[DType.float32, GPU_EVAL_BATCH, NUM_CLASSES](ctx)

    var indices_dev = DeviceMatrix[DType.int32, TRAIN_SPLIT, 1](ctx)
    var epoch_loss_dev = DeviceMatrix[DType.float32, 1, 1](ctx, fill=0.0)
    var epoch_correct_dev = DeviceMatrix[DType.int32, 1, 1](ctx, fill=0)
    var val_correct_dev = DeviceMatrix[DType.int32, 1, 1](ctx, fill=0)
    var val_preds_dev = DeviceMatrix[DType.int32, VAL_SPLIT, 1](ctx)
    var test_preds_dev = DeviceMatrix[DType.int32, TEST_ROWS, 1](ctx)

    var indices = List[Int32](capacity=TRAIN_SPLIT)
    for i in range(TRAIN_SPLIT):
        indices.append(Int32(i))

    print(
        t"\nTraining 3-layer MLP ({IMG_PIXELS} -> {HIDDEN1} -> {HIDDEN2} ->"
        t" {NUM_CLASSES}) on GPU for {EPOCHS} epochs..."
    )

    var b1_pow = Float64(1.0)
    var b2_pow = Float64(1.0)
    var step_num = 0
    var best_val_correct = -1

    var train_start = perf_counter()
    for epoch in range(1, EPOCHS + 1):
        var ep_start = perf_counter()
        shuffle(indices)
        indices_dev.copy_from_host(Span(indices))
        epoch_loss_dev.fill(0.0)
        epoch_correct_dev.fill(0)

        for b_idx in range(TRAIN_NUM_BATCHES):
            step_num += 1
            b1_pow *= BETA1
            b2_pow *= BETA2
            var lr = cosine_lr(step_num)
            var inv_bc1 = Float32(1.0 / (1.0 - b1_pow))
            var inv_bc2 = Float32(1.0 / (1.0 - b2_pow))

            var batch_start = b_idx * BATCH_SIZE
            gpu_gather_batch_rows_shifted(
                ctx,
                x_train_dev,
                y_train_dev,
                indices_dev,
                batch_start,
                UInt32(step_num * 1664525 + 1013904223),
                x_b,
                labels_dev,
            )

            # Forward + inverted dropout
            gpu_matmul_nn_bias_row_relu(ctx, x_b, w1, b1, a1)
            gpu_apply_dropout(
                ctx,
                a1,
                UInt32(step_num * 2654435761),
                DROPOUT_KEEP,
                DROPOUT_INV_KEEP,
            )

            gpu_matmul_nn_bias_row_relu(ctx, a1, w2, b2, a2)
            gpu_apply_dropout(
                ctx,
                a2,
                UInt32(step_num * 2246822519 + 1),
                DROPOUT_KEEP,
                DROPOUT_INV_KEEP,
            )

            gpu_matmul_nt_bias_row(ctx, a2, w3, b3, z3)

            # Softmax + label-smoothed CE loss, scaled dZ3/dB3, and accuracy
            gpu_softmax_ce_grad_metrics_smooth_db_rows(
                ctx,
                z3,
                labels_dev,
                dz3,
                db3,
                epoch_loss_dev,
                epoch_correct_dev,
                LABEL_SMOOTHING,
            )

            # Backward pass
            gpu_matmul_tn(ctx, dz3, a2, dw3)

            gpu_matmul_nn(ctx, dz3, w3, dz2)
            gpu_apply_relu_mask_scale(ctx, dz2, a2, DROPOUT_INV_KEEP)
            gpu_sum_axis0_scaled(ctx, dz2, Float32(1.0), db2)
            gpu_matmul_tn(ctx, a1, dz2, dw2)

            gpu_matmul_nt(ctx, dz2, w2, dz1)
            gpu_apply_relu_mask_scale(ctx, dz1, a1, DROPOUT_INV_KEEP)
            gpu_sum_axis0_scaled(ctx, dz1, Float32(1.0), db1)
            gpu_matmul_tn(ctx, x_b, dz1, dw1)

            # Fused AdamW update
            gpu_adamw_update_mlp3(
                ctx,
                w1,
                dw1,
                mw1,
                vw1,
                b1,
                db1,
                mb1,
                vb1,
                w2,
                dw2,
                mw2,
                vw2,
                b2,
                db2,
                mb2,
                vb2,
                w3,
                dw3,
                mw3,
                vw3,
                b3,
                db3,
                mb3,
                vb3,
                lr,
                inv_bc1,
                inv_bc2,
                WEIGHT_DECAY,
            )

        var val_correct = evaluate_gpu[VAL_SPLIT, True](
            ctx,
            x_val_dev,
            y_val_dev,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            a1_eval,
            a2_eval,
            z3_eval,
            val_preds_dev,
            val_correct_dev,
        )
        if val_correct > best_val_correct:
            best_val_correct = val_correct
            w1.copy_to_host(best_hw1.data)
            b1.copy_to_host(best_hb1.data)
            w2.copy_to_host(best_hw2.data)
            b2.copy_to_host(best_hb2.data)
            w3.copy_to_host(best_hw3.data)
            b3.copy_to_host(best_hb3.data)

        var avg_loss: Float32
        with epoch_loss_dev.buffer.map_to_host() as host_l:
            avg_loss = host_l[0] * INV_NUM_BATCHES

        var epoch_correct: Int
        with epoch_correct_dev.buffer.map_to_host() as host_c:
            epoch_correct = Int(host_c[0])

        var train_acc = Float32(epoch_correct) * INV_TRAIN_SPLIT_PCT
        var val_acc = Float32(val_correct) * INV_VAL_SPLIT_PCT
        var ep_time = perf_counter() - ep_start
        print(
            t"Epoch {epoch}/{EPOCHS} | loss={avg_loss} | train_acc={train_acc}%"
            t" | val_acc={val_acc}% | time={ep_time}s"
        )

    var total_train_time = perf_counter() - train_start
    print(t"\nTraining finished in {total_train_time}s!")

    w1.copy_from_host(Span(best_hw1.data))
    b1.copy_from_host(Span(best_hb1.data))
    w2.copy_from_host(Span(best_hw2.data))
    b2.copy_from_host(Span(best_hb2.data))
    w3.copy_from_host(Span(best_hw3.data))
    b3.copy_from_host(Span(best_hb3.data))

    var final_val_correct = evaluate_gpu[VAL_SPLIT, True](
        ctx,
        x_val_dev,
        y_val_dev,
        w1,
        b1,
        w2,
        b2,
        w3,
        b3,
        a1_eval,
        a2_eval,
        z3_eval,
        val_preds_dev,
        val_correct_dev,
    )
    var final_val_acc = Float32(final_val_correct) * INV_VAL_SPLIT_PCT
    print(
        t"Best Validation Accuracy: {final_val_correct}/{VAL_SPLIT}"
        t" ({final_val_acc}%)"
    )

    var val_preds = List[Int](length=TOTAL_TRAIN_ROWS, fill=0)
    with val_preds_dev.buffer.map_to_host() as host_vp:
        for i in range(VAL_SPLIT):
            val_preds[TRAIN_SPLIT + i] = Int(host_vp[i])

    print_sample_predictions(
        train_ds,
        Span(val_preds),
        TRAIN_SPLIT,
        4,
        "Validation Sample Predictions",
    )

    _ = evaluate_gpu[TEST_ROWS, False](
        ctx,
        x_test_dev,
        y_test_dummy,
        w1,
        b1,
        w2,
        b2,
        w3,
        b3,
        a1_eval,
        a2_eval,
        z3_eval,
        test_preds_dev,
        val_correct_dev,
    )
    var test_preds = List[Int](length=TEST_ROWS, fill=0)
    with test_preds_dev.buffer.map_to_host() as host_tp:
        for i in range(TEST_ROWS):
            test_preds[i] = Int(host_tp[i])

    write_submission_csv(PREDICTIONS_FILEPATH, Span(test_preds))
    print(t"\nWrote {TEST_ROWS} test predictions to {PREDICTIONS_FILEPATH}")

    print_sample_predictions(
        test_ds,
        Span(test_preds),
        0,
        4,
        "Test Set Predictions (digit-recognizer/test.csv)",
    )

    save_weights_mlp3(
        WEIGHTS_FILEPATH,
        best_hw1,
        best_hb1,
        best_hw2,
        best_hb2,
        best_hw3,
        best_hb3,
    )
    print(t"Saved trained weights to {WEIGHTS_FILEPATH}")


def main() raises:
    seed(42)
    print("Loading MNIST datasets...")
    var load_start = perf_counter()
    var train_ds = load_mnist_csv[has_label=True](
        TRAIN_FILEPATH, num_features=IMG_PIXELS, expected_rows=TOTAL_TRAIN_ROWS
    )
    var test_ds = load_mnist_csv[has_label=False](
        TEST_FILEPATH, num_features=IMG_PIXELS, expected_rows=TEST_ROWS
    )
    var load_time = perf_counter() - load_start
    print(
        t"Loaded {train_ds.num_samples} training images and"
        t" {test_ds.num_samples} test images in {load_time}s."
    )

    comptime if USE_GPU:
        train_and_predict_gpu(train_ds, test_ds)
    else:
        train_and_predict_cpu(train_ds, test_ds)
