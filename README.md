# MNIST in Pure Mojo

A complete handwritten-digit recognition pipeline written in pure Mojo and MAX—covering parallel CSV dataset loading, CPU (multi-threaded SIMD) and GPU (`max.gpu` + `TileTensor`) training, binary model serialization, and single-image console inference.

---

## How This Project Was Created

This repository began as an early Mojo experiment comparing a pure-Mojo MNIST implementation against Python/NumPy (`notes.md`, `LayoutTensor` prototypes, and kernel experiments). As Mojo and MAX evolved, those early prototypes were refactored and completed into an end-to-end training and inference project targeting current Mojo + MAX (`1.x` nightly via Pixi):

1. **Syntax & Stdlib Modernization**: Migrated legacy traits (`comptime`, `def`), `std.*` modules, `max.algorithm.parallelize`, `max.gpu`, and `layout.TileTensor` (with `UntrackedOrigin` inside device lambdas).
2. **Parallel Data Pipeline**: Replaced broken row-parsing logic with a CRLF-safe, zero-allocation byte scanner (`src/load_file.mojo`) that parses 42,000 training rows and 28,000 test rows in ~0.3s.
3. **Dual Compute Backends (CPU & GPU)**:
   - Built a multi-threaded SIMD CPU backend (`src/cpu_ops.mojo`) using `CPUMatrix`.
   - Built a `max.gpu` kernel suite (`src/gpu_ops.mojo` + `src/gpu_mem.mojo`) supporting both Linux GPUs and macOS Apple Silicon (`metal` via Xcode `MetalToolchain`).
   - Eliminated per-batch host↔device synchronization in the GPU training loop by keeping the dataset GPU-resident and fusing cross-entropy gradient, loss accumulation, and accuracy counting into single-launch kernels.
4. **Standalone Inference (`predict`)**: Added compact binary weight serialization (`src/weights.mojo`), MNIST-style image preprocessing (`src/image_io.mojo`), a console predictor (`src/predict.mojo`), unit tests (`tests/test_mnist_ops.mojo`), and sample digit images in `examples/`.

---

## Architecture & Implementation Details

### Model & Training Configuration

Defined in [`src/train.mojo`](src/train.mojo):

- **Network**: 2-layer MLP `784 -> 128 (ReLU) -> 10 (Softmax)`
- **Weight Initialization**: He initialization (`scale = sqrt(2.0 / fan_in)`), zero biases
- **Loss & Optimizer**: Cross-entropy loss with mini-batch SGD + momentum (`BATCH_SIZE = 100`, `EPOCHS = 15`, `LEARNING_RATE = 0.05`, `MOMENTUM = 0.9`)
- **Dataset Split**: First `40,000` rows of `digit-recognizer/train.csv` for training, last `2,000` rows for validation, plus `28,000` unlabeled rows from `digit-recognizer/test.csv`
- **Outputs**:
  - `predictions.csv` — Kaggle Digit Recognizer submission format (`ImageId,Label`)
  - `mnist_weights.bin` — Saved binary weights (`~398 KB`, git-ignored via `*.bin`)

### Repository Layout

| Path | Purpose |
| :--- | :--- |
| [`pixi.toml`](pixi.toml) | Pixi workspace (`mojo = "*"`, `max = "*"` from `max-nightly`) and tasks (`train`, `train-gpu`, `predict`, `test`) |
| [`src/train.mojo`](src/train.mojo) | End-to-end CPU/GPU training, validation, Kaggle test prediction CSV export, and weight saving |
| [`src/load_file.mojo`](src/load_file.mojo) | Parallel CRLF-safe CSV loader (`load_mnist_csv`) normalizing uint8 pixels to `[0.0, 1.0]` `Float32` |
| [`src/cpu_ops.mojo`](src/cpu_ops.mojo) | `CPUMatrix` and parallel/SIMD forward, backward, softmax CE, reduction, and SGD momentum ops |
| [`src/gpu_mem.mojo`](src/gpu_mem.mojo) | `DeviceMatrix[dtype, rows, cols]` owning `DeviceBuffer` with host↔device copies, `fill`, and `init_he` |
| [`src/gpu_ops.mojo`](src/gpu_ops.mojo) | GPU kernels (`matmul_nn_kernel`, `matmul_nt_kernel`, `matmul_tn_kernel`, `add_bias_relu_kernel`, `softmax_col_kernel`, `batch_ce_grad_metrics_kernel`, `eval_argmax_count_kernel`, `gather_batch_cols_kernel`, `gather_eval_cols_kernel`, `sgd_momentum_kernel`, etc.) |
| [`src/weights.mojo`](src/weights.mojo) | Binary `MNST` serializer/deserializer (`save_weights`, `load_weights`) |
| [`src/image_io.mojo`](src/image_io.mojo) | Native 8/24/32-bit BMP & P2/P5 PGM decoders, macOS `/usr/bin/sips` converter, and MNIST preprocessing |
| [`src/predict.mojo`](src/predict.mojo) | CLI tool that loads weights + an image, prints a 28×28 ASCII preview, predicted digit, and class probabilities |
| [`src/image.mojo`](src/image.mojo) | 28×28 grayscale ASCII renderer (`render_ascii_image`) and `MNISTDataset` display helpers |
| [`src/data_traits.mojo`](src/data_traits.mojo) | Shared `Displayable` / `MatrixLike` traits |
| [`tests/test_mnist_ops.mojo`](tests/test_mnist_ops.mojo) | `TestSuite` unit tests covering matrix ops, ReLU/softmax/CE, CSV CRLF parsing, weight round-trips, and PGM preprocessing/inference |
| [`examples/`](examples/) | Sample test images (`digit_0.pgm`, `digit_2.png`, `digit_3.png`, `digit_7.png`, `digit_9.bmp`) |

### GPU Execution Design

When compiled with `-D USE_GPU=true` (`pixi run train-gpu`), [`train_and_predict_gpu`](src/train.mojo) keeps the entire training loop on device:

- **One-time dataset upload**: `x_train_dev`, `y_train_dev`, `x_val_dev`, `y_val_dev`, and `x_test_dev` are uploaded once before training starts.
- **Per-epoch index shuffle**: Only a `40,000 × Int32` shuffled index array is uploaded once per epoch; `gpu_gather_batch_cols` gathers each 100-column mini-batch directly on the GPU.
- **Fused metrics & gradients**: `batch_ce_grad_metrics_kernel` computes softmax cross-entropy gradients (`dz2`), accumulates batch loss, and counts correct predictions in one kernel using `Atomic.fetch_add`.
- **Single sync per epoch**: The host reads only two scalars (`epoch_loss_dev`, `epoch_correct_dev`) once per epoch rather than copying batch buffers or labels back and forth.

### Arbitrary Image Preprocessing (`image_io.mojo`)

Real-world images rarely match MNIST's 28×28 white-on-black centering. Before feeding an image to [`predict_single`](src/predict.mojo), [`preprocess_for_mnist`](src/image_io.mojo) runs:

1. **Decoding**: Reads `.bmp` (8/24/32-bit uncompressed) and `.pgm` (P2 ASCII / P5 binary) natively in Mojo; on macOS, converts `.png`, `.jpg`, `.jpeg`, `.heic`, `.tiff`, `.tif`, and `.gif` to grayscale/BMP via `/usr/bin/sips`.
2. **Auto-Polarity Inversion**: Samples border pixels and automatically inverts light-background images (black ink on white paper) to MNIST's dark-background convention.
3. **Contrast Stretch & Noise Floor**: Rescales peak intensity to `1.0` and zeroes pixels `< 0.12`.
4. **MNIST Centering**: Crops to the foreground bounding box (`> 0.15`), scales the longest side to `20 px` while preserving aspect ratio (area-box averaging when downsampling >1.5×, bilinear interpolation otherwise), and translates the center of mass to `(13.5, 13.5)` on a `28×28` canvas.

### Observed Benchmarks (Apple M1, 15 Epochs)

| Mode | Command | Total Train Time | Final Validation Accuracy |
| :--- | :--- | :--- | :--- |
| CPU (multi-threaded SIMD) | `pixi run train` | ~11.70s | 97.40% (`1948/2000`) |
| GPU (Apple M1 `metal`) | `pixi run train-gpu` | ~3.76s (~0.25s/epoch) | 97.55% (`1951/2000`) |

---

## How to Use

### 1. Prerequisites

- [Pixi](https://pixi.sh/) installed (`pixi install` runs automatically on task invocation).
- **For training**: Place Kaggle's Digit Recognizer dataset (`train.csv` and `test.csv`) inside `digit-recognizer/`.
- **For Apple Silicon GPU training** (macOS): Install Xcode's Metal toolchain:
  ```bash
  xcodebuild -downloadComponent MetalToolchain
  ```
  *(On Linux with a supported GPU, `USE_GPU` defaults to `true` when `has_accelerator()` is true.)*

### 2. Train the Model

Run on Apple M1 GPU (or Linux GPU):
```bash
pixi run train-gpu
```

Or run on CPU:
```bash
pixi run train
```

This trains for 15 epochs, prints validation accuracy and ASCII sample predictions, writes `predictions.csv`, and saves trained weights to `mnist_weights.bin`.

### 3. Predict on Any Picture

Once `mnist_weights.bin` exists, run inference on an image:
```bash
pixi run predict examples/digit_2.png
```

You can also try the other included sample images or pass a custom weights path:
```bash
pixi run predict examples/digit_0.pgm
pixi run predict examples/digit_3.png
pixi run predict examples/digit_7.png
pixi run predict examples/digit_9.bmp

# Custom weights file:
pixi run predict path/to/image.png path/to/custom_weights.bin
```

Example output:
```text
Input image: examples/digit_2.png (140x140)
... (28x28 ASCII preview) ...
Predicted digit: 2 (confidence: 99.99421%)
Class probabilities:
  0: ...
  2: 99.99421%
  ...
```

### 4. Run Unit Tests

```bash
pixi run test
```

Runs 7 unit tests in [`tests/test_mnist_ops.mojo`](tests/test_mnist_ops.mojo) covering matrix multiplications, ReLU/bias, softmax + cross-entropy, axis reductions, CRLF CSV parsing, binary weight round-trips, and PGM preprocessing/prediction.

---

## How This Implementation Can Be Enhanced

1. **Convolutional Architecture (97.5% → 99%+ Accuracy)**
   - Replace or precede the `784 -> 128 -> 10` MLP with a small CNN (e.g., `Conv2D(1->16, 3x3) -> ReLU -> MaxPool -> Conv2D(16->32, 3x3) -> ReLU -> MaxPool -> Linear`). This handles translated/tilted handwritten strokes much better than a flat MLP.
2. **Training Recipe & Checkpointing**
   - Save the **best validation accuracy** checkpoint rather than always saving epoch 15 weights.
   - Add Adam / AdamW and a learning-rate decay schedule (e.g., cosine annealing or step decay) instead of fixed `LEARNING_RATE = 0.05`.
   - Add light on-the-fly data augmentation during training (random ±2px shifts, slight rotations/scaling) to raise confidence on real-world photos and tricky digits (like `examples/digit_7.png`).
3. **Pure-Mojo PNG/JPEG Decoding**
   - Currently `.png`/`.jpg` decoding relies on macOS `/usr/bin/sips` in `src/image_io.mojo`, while `.bmp` and `.pgm` are pure Mojo. Adding a pure-Mojo DEFLATE/zlib + PNG chunk decoder (or PPM/stb-style loader) would make `pixi run predict` on `.png` files completely cross-platform on Linux as well.
4. **Generalize Tail Batch Sizes & Multi-Digit Inference**
   - `evaluate_cpu` and `evaluate_gpu` currently iterate `num_samples // BATCH_SIZE` full batches of 100 (exact for 2,000 and 28,000). Handling remainder tail elements makes evaluation work on arbitrary sample counts.
   - Add a batch image CLI or connected-component segmentation in `src/predict.mojo` to recognize multiple digits in a single photo.
5. **GPU Kernel Optimization & MAX Graph Integration**
   - Replace the naive element-wise matmul thread mapping in `src/gpu_ops.mojo` with shared-memory tiled matmul (`block_dim` tiles + Tensor Core / MMA paths on supported hardware) and benchmark against high-level `max.nn` / MAX Graph ops.
