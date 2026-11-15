# How `mnist_mojo` Works: From Basic Math to GPU Neural Networks

This guide explains **everything** inside this project from the ground up. It is written for someone who is comfortable with high-school and early-college mathematics (algebra, functions, summation notation, exponentials/logarithms, and basic derivatives/chain rule), without assuming prior knowledge of **matrices**, **machine learning**, **CPU hardware optimization**, or **GPU programming**.

---

## Table of Contents

1. [The Goal & Project Map](#1-the-goal--project-map)
2. [Matrix Operations Explained From Scratch](#2-matrix-operations-explained-from-scratch)
3. [What the Neural Network Actually Is](#3-what-the-neural-network-actually-is)
4. [How the Network Learns (Calculus & Backpropagation)](#4-how-the-network-learns-calculus--backpropagation)
5. [Memory Layout: Why Data Arrangement Determines Speed](#5-memory-layout-why-data-arrangement-determines-speed)
6. [How CPU Acceleration Works (`src/cpu_ops.mojo`)](#6-how-cpu-acceleration-works-srccpu_opsmojo)
7. [How a GPU Works & How Our GPU Kernels Run (`src/gpu_ops.mojo`)](#7-how-a-gpu-works--how-our-gpu-kernels-run-srcgpu_opsmojo)
8. [End-to-End Pipeline: Data, Weights, Prediction & Tests](#8-end-to-end-pipeline-data-weights-prediction--tests)

---

## 1. The Goal & Project Map

The problem is classic handwritten-digit recognition (**MNIST**):

- **Input**: A $28 \times 28$ grayscale image of a handwritten digit (`0` through `9`). Unrolled row by row, that is a list of $28 \times 28 = 784$ numbers, each scaled into $[0.0, 1.0]$ where `0.0` is black background and `1.0` is white ink.
- **Output**: 10 probabilities $(p_0, p_1, \dots, p_9)$ that sum to $1.0$, where $p_d$ is how confident the model is that the drawing is digit $d$.

Instead of writing hand-crafted rules ("if there is a loop at the top and a vertical line, it might be a 9"), we build a parameterized mathematical function $f_\theta(x)$ with ~101,770 tunable numbers ($\theta$) and use calculus to adjust those numbers automatically from 40,000 labeled examples.

### Project File Map

| File | Responsibility |
| :--- | :--- |
| [`src/train.mojo`](src/train.mojo) | Orchestrates the training loop (`train_and_predict_cpu` / `train_and_predict_gpu`), validation, test-set submission (`predictions.csv`), and saving weights (`mnist_weights.bin`). |
| [`src/cpu_ops.mojo`](src/cpu_ops.mojo) | `CPUMatrix`, zero-overhead register views (`MatrixView`, `StaticMatrixView`), and SIMD + multi-threaded CPU math kernels. |
| [`src/gpu_mem.mojo`](src/gpu_mem.mojo) | `DeviceMatrix` wrapper around GPU buffers (`DeviceBuffer`) and host/device memory transfers. |
| [`src/gpu_ops.mojo`](src/gpu_ops.mojo) | GPU kernels for tiled matrix multiplication, fused softmax/cross-entropy, coalesced reductions, and fused SGD+momentum updates. |
| [`src/load_file.mojo`](src/load_file.mojo) | Fast SIMD + parallel CSV parser (`load_mnist_csv`) that reads pixel integers and divides by `255.0`. |
| [`src/weights.mojo`](src/weights.mojo) | Saves and loads trained parameters in a compact binary format (`MNST` header + raw `Float32` arrays). |
| [`src/image_io.mojo`](src/image_io.mojo) & [`src/image.mojo`](src/image.mojo) | Loads real-world images (`.png`, `.jpg`, `.bmp`, `.pgm`, etc.), converts to grayscale, centers/scales to $28 \times 28$ MNIST style, and renders ASCII previews. |
| [`src/predict.mojo`](src/predict.mojo) | CLI tool that loads `mnist_weights.bin`, preprocesses one image, and prints the predicted digit and class probabilities. |
| [`tests/test_mnist_ops.mojo`](tests/test_mnist_ops.mojo) | Unit tests verifying matrix ops, activations, softmax/cross-entropy, CSV parsing, weight roundtrips, and prediction. |

---

## 2. Matrix Operations Explained From Scratch

If you know summation ($\sum$), you already understand almost all of matrix math—matrices are just a compact bookkeeping notation for doing **many weighted sums simultaneously**.

### 2.1 Vectors and Weighted Sums (Dot Product)

Suppose one image has 784 pixel values $x = (x_0, x_1, \dots, x_{783})$.
Suppose a "neuron" wants to measure whether those pixels resemble a particular stroke pattern. It assigns an importance **weight** $w_k$ to each pixel $x_k$, and computes a single score:

$$z = \sum_{k=0}^{783} x_k \cdot w_k + b$$

- Positive $w_k$ means "bright ink at pixel $k$ supports this pattern."
- Negative $w_k$ means "bright ink at pixel $k$ argues against this pattern."
- $b$ (the **bias**) is just a constant offset—like the $y$-intercept in $y = mx + b$—that shifts how easily the neuron activates even when pixels are dim.

This sum $\sum_k x_k w_k$ is called a **dot product**.

### 2.2 What a Matrix Is in Math and in Memory

A **matrix** of shape $(M \times K)$ is simply a rectangular grid with $M$ rows and $K$ columns.
In computer memory, RAM is a 1D strip of bytes, not a 2D grid. In this codebase (`CPUMatrix` and `DeviceMatrix` with `row_major`), we store a $(M \times K)$ matrix in **row-major order**: row 0 comes first (`K` numbers), immediately followed by row 1 (`K` numbers), and so on.

So the element at row $r$ and column $c$ is stored at 1D index:

$$\text{index}(r, c) = r \cdot K + c$$

Why does that matter? Because moving across a **row** ($c \to c+1$) steps by `1` in memory (contiguous neighbors), whereas moving down a **column** ($r \to r+1$) jumps by `K` elements (strided).

### 2.3 Matrix Multiplication (`C = A @ B`)

Now suppose we don't just have 1 image and 1 neuron—we have a **mini-batch** of $M = 100$ images (each with $K = 784$ pixels) and $N = 128$ neurons (each with 784 weights).

- Pack the 100 images as rows of matrix $A$ with shape $(100 \times 784)$.
- Pack the 128 neurons' weight lists as columns of matrix $B$ with shape $(784 \times 128)$.

**Matrix multiplication** $C = A \cdot B$ produces a $(100 \times 128)$ matrix where entry $C_{i, j}$ is the dot product of row $i$ of $A$ (image $i$) with column $j$ of $B$ (neuron $j$):

$$C_{i, j} = \sum_{k=0}^{K-1} A_{i, k} \cdot B_{k, j}$$

Notice the shape rule: $(M \times K) \cdot (K \times N) \to (M \times N)$. The inner dimension $K$ must match because that is the length of the dot product being summed over.

### 2.4 Transposes and Why We Have `NN`, `NT`, and `TN` Variants

The **transpose** of a matrix $M$, written $M^T$, flips rows and columns:

$$(M^T)_{r, c} = M_{c, r}$$

If $M$ has shape $(R \times C)$, then $M^T$ has shape $(C \times R)$.

During training (especially backpropagation), we frequently need to multiply by a transposed matrix—for example $A \cdot B^T$ or $A^T \cdot B$. If we physically copied and rearranged memory every time we needed a transpose, we would waste memory and bandwidth. Instead, both `cpu_ops.mojo` and `gpu_ops.mojo` provide three specialized multiplication routines:

| Variant | Formula | Input Shapes | Output Shape |
| :--- | :--- | :--- | :--- |
| **`matmul_nn`** (Normal-Normal) | $C_{i, j} = \sum_k A_{i, k} B_{k, j}$ | $A: (M, K),\; B: (K, N)$ | $(M, N)$ |
| **`matmul_nt`** (Normal-Transpose) | $C_{i, j} = \sum_k A_{i, k} B_{j, k}$ | $A: (M, K),\; B: (N, K)$ | $(M, N)$ |
| **`matmul_tn`** (Transpose-Normal) | $C_{i, j} = \sum_k A_{k, i} B_{k, j}$ | $A: (K, M),\; B: (K, N)$ | $(M, N)$ |

Look closely at **`matmul_nt`** ($A \cdot B^T$): because $B$ is $(N, K)$ in memory, row $j$ of $B$ is stored **contiguously** in memory, just like row $i$ of $A$! That is why our second layer stores $W_2$ with shape `(10, 128)` and computes $A_1 W_2^T$ using `matmul_nt`.

---

## 3. What the Neural Network Actually Is

Our network in `src/train.mojo` is a **2-Layer Multi-Layer Perceptron (MLP)** with architecture **`784 -> 128 -> 10`**. It is just a composition of two linear transformations separated by one simple non-linear function, followed by Softmax.

### 3.1 Layer 1: Feature Detection + ReLU

Given a batch of $B = 100$ images $X_b$ of shape $(100, 784)$, first-layer weights $W_1$ of shape $(784, 128)$, and bias vector $b_1$ of length $128$:

$$Z_1 = X_b W_1 + b_1 \quad \text{(bias } b_1[j] \text{ is added to every image in column } j\text{)}$$

$$A_1 = \text{ReLU}(Z_1) = \max(0, Z_1)$$

**Why do we need ReLU?**
Without a non-linear function like $\max(0, z)$, stacking two linear matrix multiplications would collapse into a single linear one:

$$(X W_1) W_2 = X (W_1 W_2) = X W_{\text{combined}}$$

A purely linear model can only draw flat hyperplanes across pixel space—it cannot combine parts like *"has a top loop AND a diagonal tail, OR a closed loop on the bottom"* the way handwritten digits require.

**ReLU (Rectified Linear Unit)** simply clamps negative values to `0` and leaves positive values untouched:
- If a hidden neuron doesn't match the image ($z \le 0$), it outputs `0` (silent).
- If it matches ($z > 0$), it passes its strength forward linearly.
- Its derivative is trivial (`1` if $z > 0$, `0` if $z \le 0$), and it doesn't shrink gradients toward zero like older S-shaped curves (`sigmoid`/`tanh`).

### 3.2 Weight Initialization: He Normal (`init_he`)

Before training starts, what should $W_1$ and $W_2$ be?
- **Cannot be all zeros**: If every neuron starts identical, every neuron receives the exact same gradient and stays identical forever (symmetry trap).
- **Cannot be too large or too small**: Too large causes values/exponentials to explode; too small causes signals to vanish.

We initialize biases $b_1, b_2$ to `0.0`, and sample weights from a Gaussian distribution with mean `0` and standard deviation:

$$\sigma = \sqrt{\frac{2}{n_{\text{in}}}}$$

This is **He Initialization** (in `CPUMatrix.init_he` and `DeviceMatrix.init_he`):
- For $W_1$: $n_{\text{in}} = 784$, so $\sigma = \sqrt{2/784} \approx 0.0505$.
- For $W_2$: $n_{\text{in}} = 128$, so $\sigma = \sqrt{2/128} = 0.125$.

Why the factor of $2$ in the numerator? Because ReLU zeros out roughly half of its inputs (negative half), cutting variance in half; multiplying variance by $2 / n_{\text{in}}$ keeps the signal's variance roughly constant across layers.

### 3.3 Layer 2: Raw Scores (Logits)

Next, we take the 128 hidden features $A_1$ $(100, 128)$ and combine them into 10 raw scores (**logits**), one per digit `0..9`, using $W_2$ $(10, 128)$ and $b_2$ $(10, 1)$:

$$Z_2 = A_1 W_2^T + b_2 \qquad \text{(shape: } 100 \times 10\text{)}$$

### 3.4 Turning Scores Into Probabilities: Softmax

Raw logits $z = (z_0, \dots, z_9)$ can be any real numbers (negative, zero, or huge). To turn them into valid probabilities $(p_0, \dots, p_9)$ that are positive and sum to $1$:

$$p_c = \frac{e^{z_c - m}}{\sum_{j=0}^{9} e^{z_j - m}}, \qquad m = \max_{j} z_j$$

**Why subtract $m = \max_j z_j$?**
Mathematically, subtracting a constant $m$ from every $z_j$ cancels out completely:

$$\frac{e^{z_c - m}}{\sum_j e^{z_j - m}} = \frac{e^{-m} e^{z_c}}{e^{-m} \sum_j e^{z_j}} = \frac{e^{z_c}}{\sum_j e^{z_j}}$$

In 32-bit floating-point arithmetic (`Float32`), $e^{89}$ overflows to `Inf`! By subtracting $m$, the largest exponent becomes $e^{0} = 1.0$ and all others are in $(0, 1]$, guaranteeing zero overflow while keeping $\sum_c p_c = 1$.

Notice that for **prediction/evaluation** (`evaluate_cpu`, `evaluate_gpu`, `cpu_eval_batch_logits_rows`, `eval_argmax_rows_count_kernel`), we don't even need to compute $e^{z_c}$! Since $e^z$ is strictly increasing, the largest probability always belongs to the largest logit:

$$\arg\max_c p_c = \arg\max_c z_c$$

Skipping `exp()` during validation and test evaluation saves computation with identical predictions.

### 3.5 Measuring Error: Cross-Entropy Loss

Suppose an image's true label is $y \in \{0, \dots, 9\}$, and our model outputs probability $p_y$ for that true digit. We define the sample's **Cross-Entropy Loss** as:

$$\mathcal{L} = -\log(\max(p_y, 10^{-7}))$$

- If the model is 99% confident in the right digit ($p_y = 0.99$), $\mathcal{L} = -\log(0.99) \approx 0.01$ (tiny error).
- If the model is only 1% confident ($p_y = 0.01$), $\mathcal{L} = -\log(0.01) \approx 4.61$ (large penalty).
- The `max(..., 1e-7)` clamp prevents $\log(0) = -\infty$.

The batch loss is the average $\frac{1}{B} \sum_{i=0}^{B-1} \mathcal{L}_i$. Training's entire purpose is to push this average loss as low as possible without memorizing noise.

---

## 4. How the Network Learns (Calculus & Backpropagation)

With ~101,770 parameters ($784 \times 128 + 128 + 10 \times 128 + 10 = 101{,}770$), how do we know whether to nudge each parameter up or down? **Derivatives!**

For any parameter $w$, its partial derivative $\frac{\partial \mathcal{L}}{\partial w}$ tells us:
- If $\frac{\partial \mathcal{L}}{\partial w} > 0$, increasing $w$ increases the loss, so we should **decrease** $w$.
- If $\frac{\partial \mathcal{L}}{\partial w} < 0$, increasing $w$ decreases the loss, so we should **increase** $w$.

### 4.1 Mini-Batch Training, Epochs, and Train/Validation Split

Our dataset (`digit-recognizer/train.csv`) has `42,000` labeled images:
- First `40,000` images (`TRAIN_SPLIT`): used to compute gradients and update weights.
- Last `2,000` images (`VAL_SPLIT`): **held out**—never used for weight updates. Checking accuracy on these unseen images after every epoch tells us whether the network is genuinely generalizing rather than memorizing.

Each **epoch** shuffles the 40,000 training indices and processes them in `400` **mini-batches** of `BATCH_SIZE = 100` images (`15` epochs total). Mini-batches are much faster than 1-image-at-a-time updates (they expose parallelism) and much faster than waiting for all 40,000 images before making a single update.

### 4.2 Backpropagation: Step-by-Step Chain Rule

Let $B = 100$ be the batch size and let's trace the derivatives backward from the loss $\mathcal{L}$ to every weight and bias:

#### Step A: Derivative Through Softmax + Cross-Entropy (`dZ2`)
For a single sample with logits $z$, softmax probabilities $p$, and true class $y$, the derivative of $\mathcal{L} = -\log(p_y)$ with respect to logit $z_c$ works out cleanly via the quotient/chain rule:

$$\frac{\partial \mathcal{L}}{\partial z_c} = \begin{cases} p_c - 1 & \text{if } c = y \\ p_c & \text{if } c \neq y \end{cases}$$

*(Why? Because $\frac{\partial (-\log p_y)}{\partial z_c} = -\frac{1}{p_y}\frac{\partial p_y}{\partial z_c}$, and $\frac{\partial p_y}{\partial z_c} = p_y(\mathbf{1}_{c=y} - p_c)$, so $p_y$ cancels out completely!)*

Notice how intuitive this is:
- For the true class $y$: gradient is $p_y - 1 \le 0$. Subtracting a negative gradient *pushes $z_y$ up*.
- For wrong classes $c \neq y$: gradient is $p_c \ge 0$. Subtracting a positive gradient *pushes $z_c$ down*.
- When the model is already 100% right ($p_y = 1$, others $0$), all gradients become $0$ and learning stops cleanly.

Let $dZ_2$ be the $(100, 10)$ matrix of $(p_c - \mathbf{1}_{c=y})$ values (and note we keep the $1/B$ averaging factor in the matrix multiplications/sums below so $dZ_2$ stays well-scaled).

#### Step B: Gradients for Layer 2 (`dW2`, `dB2`)
Since $Z_2[b, c] = \sum_{h} A_1[b, h] W_2[c, h] + b_2[c]$, the chain rule gives:

$$dW_2 = \frac{1}{B} \, dZ_2^T A_1 \qquad \text{shape: } (10, 100) \cdot (100, 128) = (10, 128)$$

$$db_2[c] = \frac{1}{B} \sum_{b=0}^{B-1} dZ_2[b, c] \qquad \text{shape: } (10, 1)$$

#### Step C: Backpropagating Into Layer 1 (`dZ1`)
First, propagate error back through $W_2$: $dA_1 = dZ_2 W_2$ (shape $(100, 10) \cdot (10, 128) = (100, 128)$).
Next, pass through ReLU $A_1 = \max(0, Z_1)$. Since the derivative of ReLU is `1` where $Z_1 > 0$ and `0` where $Z_1 \le 0$, and since $A_1 > 0$ holds at the **exact same entries** as $Z_1 > 0$:

$$dZ_1 = (dZ_2 W_2) \odot \mathbf{1}[A_1 > 0]$$

where $\odot$ means elementwise multiplication. Notice this clever memory saving: because $A_1 > 0 \iff Z_1 > 0$, we never need to allocate or keep a separate $Z_1$ buffer!

#### Step D: Gradients for Layer 1 (`dW1`, `dB1`)
Since $Z_1 = X_b W_1 + b_1$:

$$dW_1 = \frac{1}{B} \, X_b^T dZ_1 \qquad \text{shape: } (784, 100) \cdot (100, 128) = (784, 128)$$

$$db_1[h] = \frac{1}{B} \sum_{b=0}^{B-1} dZ_1[b, h] \qquad \text{shape: } (128, 1)$$

### 4.3 Parameter Update: SGD with Momentum

With all four gradients ($dW_1, db_1, dW_2, db_2$) computed, we update each parameter $\theta$ using **Stochastic Gradient Descent with Momentum** (`LEARNING_RATE = 0.05`, `MOMENTUM = 0.9`):

$$v \leftarrow \mu \cdot v + g$$

$$\theta \leftarrow \theta - \eta \cdot v$$

where $\eta = 0.05$ is the learning rate, $\mu = 0.9$ is momentum, $g$ is the gradient, and $v$ is a persistent velocity buffer initialized to `0`.

**Why momentum?** Mini-batches of 100 images have random sampling noise, making raw gradients jitter from step to step. Maintaining an exponentially weighted moving velocity $v$ acts like a heavy ball rolling down a valley: oscillations across steep walls cancel out, while consistent downhill directions build speed.

---

## 5. Memory Layout: Why Data Arrangement Determines Speed

Mathematically, transposing all matrices in a formula gives the exact same numbers. Hardware-wise, **memory layout can double or halve speed**.

### Why Sample-Major `(batch, features)` Wins

In `mnist_mojo`, dataset images are stored on disk and in memory as consecutive rows of 784 pixels: sample 0 (`0..783`), sample 1 (`784..1567`), etc.

We organize **all activations** in **sample-major** `(batch, features)` order and `w1` as `(784, 128)`:

| Tensor | Shape | Meaning |
| :--- | :--- | :--- |
| `x_b` | `(100, 784)` | 100 training images, each a contiguous 784-float row |
| `w1` | `(784, 128)` | 784 input rows $\times$ 128 hidden columns |
| `a1`, `dz1` | `(100, 128)` | 100 samples $\times$ 128 hidden features |
| `w2` | `(10, 128)` | 10 output rows $\times$ 128 hidden features |
| `z2`, `dz2` | `(100, 10)` | 100 samples $\times$ 10 class logits/gradients |

Why this specific layout is so fast on both CPU and GPU:

1. **Contiguous batch gather (`cpu_gather_batch_rows` / `gpu_gather_batch_rows`)**:
   - Gathering a shuffled training batch copies 100 contiguous 784-float slices (`unsafe_memcpy` on CPU; coalesced `BLOCK_1D` strides along `feat` on GPU).
2. **Zero-copy evaluation (`evaluate_cpu` / `evaluate_gpu`)**:
   - Validation (`2,000` images) and test (`28,000` images) are already contiguous in `dataset.pixels` / `x_val_dev` / `x_test_dev`.
   - CPU creates a zero-copy `StaticMatrixView[500, 784]` pointing directly at `pixels_ptr + offset * 784`.
   - GPU passes `a_row_offset = offset` directly to `gpu_matmul_nn_offset_bias_row_relu` in `2,000`-image batches (1 pass for all of validation!). Zero gather buffer, zero copy.
3. **Vectorized `dW1` via `matmul_tn`**:
   - $dW_1 = \frac{1}{B} X_b^T dZ_1$ has shape `(784, 100) @ (100, 128)`. Both inner loads along `784` (`i`) and `128` (`j`) are contiguous in memory, enabling SIMD vector loads on CPU and coalesced tile loads on GPU.
4. **Coalesced `dB1` and fused `dB2`**:
   - `db1` is a column sum over `(100, 128)` (`sum_axis0`), so every thread/SIMD lane walks down rows while advancing contiguously across the 128 columns.
   - `db2` is fused directly inside the Softmax+CE pass (`cpu_softmax_ce_grad_metrics_rows` and `gpu_softmax_ce_grad_metrics_db_rows_kernel`), removing an entire pass and 6,000 GPU kernel launches over 15 epochs.
5. **Backward compatibility with `mnist_weights.bin`**:
   - `predict.mojo` expects `w1` to have shape `(128, 784)`. Right before `save_weights`, `train.mojo` runs `cpu_transpose_static` once to transpose `(784, 128) -> (128, 784)` so saved weights remain 100% compatible.

---

## 6. How CPU Acceleration Works (`src/cpu_ops.mojo`)

Modern CPUs are fast only when you feed their **vector units**, keep hot data in **L1/L2 cache**, and avoid **thread synchronization overhead**. Here is how `cpu_ops.mojo` achieves ~0.09s per epoch for 40,000 images:

### 6.1 `TrivialRegisterPassable` Views & `comptime` Specialization

- **`CPUMatrix`** owns a heap-allocated `List[Float32]`.
- **`StaticMatrixView[rows, cols]`** is a tiny struct holding just a single raw pointer (`ptr`), annotated with `TrivialRegisterPassable`. It has zero heap allocation, zero reference counting, and fits directly in a CPU register.
- Because `rows` and `cols` are **`comptime` parameters** (known when Mojo compiles the binary), the compiler:
  - Eliminates tail loops whenever `N % SIMD_W == 0` (true for `N = 128` and `K = 784`!).
  - Unrolls short inner loops completely when `K_s <= 16` (e.g., `K = 10` in `dZ1 = dZ2 @ W2`).
  - Prunes unused epilogue branches (`comptime if epilogue == ...`) so there is zero `if/else` branching inside inner loops.

### 6.2 SIMD Vectorization & FMA

Instead of multiplying and adding one `Float32` at a time, **SIMD** (Single Instruction, Multiple Data) packs `SIMD_W` floats (4 floats on Apple Silicon NEON / 128-bit registers, or 8/16 on AVX2/AVX-512) into a single vector register and executes **FMA** (Fused Multiply-Add: `av.fma(bv, acc)` computes `a * b + acc` in one instruction).

### 6.3 L1-Cache Strip Tiling & `4x4` Register Micro-Kernels

Consider `Fwd1`: `X_b (100, 784) @ W1 (784, 128)`:
- If we computed one row of `A1` at a time, we would re-read all of `W1` (`784 * 128 * 4 = 401 KB`) from cache 100 times.
- Instead, `_matmul_nn_impl` uses a **`j`-outer strip-tiled schedule** and **`4x4` register micro-kernels** (`_matmul_nn_strip4x4`):
  - It fixes a strip of `SIMD_4W` columns (`16` floats wide) of `W1` in L1 cache and streams 4 rows of `X_b` at a time (`i, i+1, i+2, i+3`).
  - Four rows $\times$ up to 4 SIMD vectors sit completely in CPU registers (`c00..c33`) for the entire $K$-loop, reusing each loaded `W1` vector 4 times before ever touching memory!
  - `_matmul_tn_strip4x4` does the same for `dW1 = X_b.T @ dZ1`, loading 4 contiguous elements of `X_b[k, i..i+3]` and 4 contiguous vectors of `dZ1[k, j..]` so every memory read is sequential.

### 6.4 Parallelism Threshold (`PARALLEL_FLOP_THRESHOLD`)

Spawning/waking worker threads (`parallelize`) costs a few microseconds. For small matrix operations like `Fwd2` (`100 x 128 x 10` $\approx 256{,}000$ FLOPs), running single-threaded is actually *faster* than waking threads. We therefore compute `flops = 2 * M * N * K` at `comptime` and only parallelize when `flops >= 500_000`.

### 6.5 Epilogue Fusion

Normally, doing `C = relu(A @ B + bias)` as separate functions would write `A @ B` to RAM, read it back to add `bias`, write it to RAM, read it back for `max(0, x)`, and write it a third time.
With **epilogue fusion** (`_store_nn_tile_row`), while the accumulated dot-product results are *still sitting in SIMD registers*, we immediately add the bias (`EPILOGUE_BIAS_ROW`), apply `max(val, 0)` (`EPILOGUE_BIAS_ROW_RELU`), or multiply by the ReLU mask (`EPILOGUE_RELU_MASK`) and store to RAM **once**.

---

## 7. How a GPU Works & How Our GPU Kernels Run (`src/gpu_ops.mojo`)

### 7.1 CPU vs. GPU Mental Model

- **CPU**: A handful of very smart, fast cores (e.g., 8–16 cores) with huge caches, optimized to run one sequence of instructions as fast as possible.
- **GPU**: Thousands of much simpler arithmetic lanes designed to execute the **same program (kernel)** on thousands of data elements **in parallel**.

In GPU programming, work is organized into a 3-level hierarchy:

1. **Grid**: All threads launched by one kernel call, split into **blocks** (`grid_dim`).
2. **Threadblock**: A group of threads (e.g. `16 x 16 = 256` threads via `block_dim`) that run on the same compute core and can share ultra-fast on-chip **Shared Memory** (`.SHARED`) and synchronize with `barrier()`.
3. **Warp / SIMD-group**: A hardware lockstep group of threads inside a block (`WARP_SIZE` threads, e.g. 32 or 64) that execute the same instruction together and can exchange register values directly using warp primitives like `warp.sum()`.

### 7.2 The GPU Memory Hierarchy

| Memory Tier | Scope | Speed | Size | How We Use It |
| :--- | :--- | :--- | :--- | :--- |
| **Registers** | Private per thread | Fastest (~1 cycle) | Tiny | Loop accumulators (`acc00, acc01, acc10, acc11`) |
| **Shared Memory (`.SHARED`)** | Shared within 1 block | Very fast (~few cycles) | ~32 KB / block | Matrix tiles `sa`, `sb` and warp reduction scratchpads |
| **Global Memory (`DeviceBuffer`)** | Visible to all blocks & host | Slowest (hundreds of cycles) | Gigabytes | `w1, w2, x_train_dev, x_b, a1, z2, ...` |

### 7.3 Coalesced Memory Access

When 32 threads in a warp read global memory at the same time, if thread `0` reads address `addr + 0`, thread `1` reads `addr + 1`, ..., thread `31` reads `addr + 31`, the GPU hardware **coalesces** all 32 requests into **1 single memory transaction**! If instead they stride across rows (e.g., jumping by `784` floats per thread), the hardware must issue 32 separate transactions.

That is why:
- `gather_batch_rows_kernel` assigns `thread_idx.x` along `feat` (`0..783`) and `block_idx.x` along batch sample `b` (`0..99`).
- `sum_axis0_scaled_kernel` assigns `global_idx.x` along `col` (`0..127`), so on every loop step `r`, all 128 threads read `m[r, 0..127]` in a single coalesced sweep with zero shared memory and zero barriers.

### 7.4 Tiled GPU Matrix Multiplication (`16x16` and `32x32` Register Tiling)

If every thread in a naive GPU matmul read its entire row of `A` and column of `B` directly from global memory, each element would be fetched from slow VRAM dozens of times.

Instead, `matmul_nn_kernel` and `matmul_tn_kernel` use **Shared-Memory Tiling**:
1. Divide `K` into chunks of `TILE_SIZE = 16`.
2. All 256 threads in the `16x16` block cooperate to load one `16x16` tile of `A` (`sa`) and one `16x16` tile of `B` (`sb`) from global memory into `.SHARED` memory (coalesced!).
3. Call `barrier()` so every thread knows the tile is ready.
4. Every thread multiplies its row of `sa` by its column of `sb` for `k = 0..15`, accumulating in a fast **register**.
5. Call `barrier()`, then move to the next `k_base` tile.

For larger matrices (`M >= 64 and N >= 64`, such as `Fwd1`, `Eval`, and `Bwd dW1`), `gpu_ops.mojo` dispatches **`matmul_nn_32x32_kernel`** and **`matmul_tn_32x32_kernel`** (`BIG_TILE = 32`):
- Still uses `16 x 16 = 256` threads per block, **but each thread computes a `2 x 2` output sub-tile** (`acc00, acc01, acc10, acc11`)!
- In the inner loop, loading 2 values from `sa` (`a0, a1`) and 2 values from `sb` (`b0, b1`) performs **4 multiply-adds** (`2x` arithmetic intensity).
- That cuts global memory traffic and `barrier()` calls in **half** compared to `16x16` blocks.

### 7.5 Shared-Memory Bank Conflict Padding (`+ 1`)

On-chip `.SHARED` memory is divided into parallel SRAM **banks** (typically 32 banks, 4 bytes wide). If multiple threads in a warp read the **same column** `sa[0..15, k]` and the row width is `16` or `32`, multiple rows map to the *exact same memory bank*, forcing the hardware to serialize those reads one after another (**bank conflict**).

By declaring shared tiles with **`+ 1` extra padding column**:

```mojo
row_major[TILE_SIZE, TILE_SIZE + 1]()   // stride 17 instead of 16
row_major[BIG_TILE, TILE_SIZE + 1]()    // stride 17 instead of 16
row_major[TILE_SIZE, BIG_TILE + 1]()    // stride 33 instead of 32
```

consecutive rows shift by 1 bank, spreading column accesses evenly across banks so reads happen in parallel with zero bank conflicts.

### 7.6 Kernel Fusion & Reductions on GPU

Launching a GPU kernel has fixed host/driver overhead (tens of microseconds per launch on Metal). Sending many tiny kernels hurts performance, so `gpu_ops.mojo` fuses operations aggressively:

- **`softmax_ce_grad_metrics_db_rows_kernel`**: In a single launch (1 block of 128 threads for `BATCH_SIZE = 100`), each thread computes softmax, cross-entropy loss, accuracy, and `dZ2` for its image in registers, then reduces batch loss, correct count, **and all 10 `db2` gradients** using fast `warp.sum()` + 1 shared-memory cross-warp reduction.
- **`eval_argmax_rows_count_kernel`**: Processes `2,000` evaluation images in a single block of 256 strided threads (`b = tid; b < 2000; b += 256`), writing predictions and reducing `total_correct` via `warp.sum()` without atomic contention.
- **`sgd_momentum_mlp_kernel`**: Updates all 4 parameter tensors (`w1`, `b1`, `w2`, `b2`) and their 4 momentum velocities (`vw1`, `vb1`, `vw2`, `vb2`) in **1 single kernel launch** instead of 4 separate launches per batch.

---

## 8. End-to-End Pipeline: Data, Weights, Prediction & Tests

1. **Data Loading (`src/load_file.mojo`)**:
   - `load_mnist_csv` reads the whole CSV file at once, scans for `\n` using byte-wide SIMD (`U8_SIMD_W`, stripping `\r` for Windows CRLF compatibility), and parses all 42,000 rows in parallel (`parallelize`), normalizing every `0..255` pixel integer by `1.0 / 255.0`.
2. **Training & Validation (`src/train.mojo`)**:
   - Runs 15 epochs on CPU (`pixi run train`) or Apple M1 Metal GPU (`pixi run train-gpu`).
   - Validation accuracy reaches ~`97.7%` (`1954–1955 / 2000`).
   - Writes Kaggle test predictions to `predictions.csv` and saves trained weights to `mnist_weights.bin`.
3. **Binary Weights Format (`src/weights.mojo`)**:
   - Stores a 16-byte little-endian header: magic bytes `"MNST"` + `Int32(hidden_size)` + `Int32(input_size)` + `Int32(num_classes)`, followed by the raw `Float32` bytes of `W1 (128x784)`, `b1 (128x1)`, `W2 (10x128)`, and `b2 (10x1)`.
4. **Predicting Custom Images (`src/image_io.mojo`, `src/predict.mojo`)**:
   - Supports `.bmp` and `.pgm` natively, and converts `.png`/`.jpg`/`.heic`/`.tiff`/`.gif` via macOS `sips` into a temporary `.bmp`.
   - Converts RGB to grayscale using standard luminance `0.299 R + 0.587 G + 0.114 B`.
   - Detects light backgrounds (`border_mean > 0.5`) and auto-inverts so ink is white on black, crops the digit's tight bounding box, bilinearly resizes it to fit a $20 \times 20$ box, and places its **center of mass** at the center of a $28 \times 28$ grid (matching how original MNIST digits were normalized).
   - Runs forward propagation (`cpu_predict_mlp_single`) with `load_weights` and prints the ASCII preview, predicted digit, and all 10 class probabilities.
5. **Verification (`tests/test_mnist_ops.mojo`)**:
   - `pixi run test` runs 7 unit tests checking `matmul` (`NN`, `NT`, `TN`), bias+ReLU, softmax+cross-entropy+argmax, `sum_axis1_scaled`, CRLF CSV parsing, `save_weights`/`load_weights` binary roundtrip, and PGM preprocessing + prediction.
