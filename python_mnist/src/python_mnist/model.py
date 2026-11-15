"""MAX neural network model and compiled training step for MNIST."""

from __future__ import annotations

from max import functional as F
from max.driver import CPU, Accelerator, Device, accelerator_count
from max.dtype import DType
from max.graph import TensorType
from max.nn import Linear, Module
from max.tensor import Tensor, default_dtype
import numpy as np

IMG_PIXELS = 784
HIDDEN1 = 512
HIDDEN2 = 256
NUM_CLASSES = 10

BETA1 = 0.9
BETA2 = 0.999
ADAM_EPS = 1e-8
WEIGHT_DECAY = 1e-4


def select_device(device_pref: str = "auto") -> Device:
    """Selects MAX execution device ('auto', 'cpu', or 'gpu')."""
    pref = device_pref.lower()
    if pref == "cpu":
        return CPU()
    if pref == "gpu":
        if accelerator_count() == 0:
            raise RuntimeError("GPU requested, but no MAX accelerator is available.")
        return Accelerator()
    if pref == "auto":
        return Accelerator() if accelerator_count() > 0 else CPU()
    raise ValueError(f"Unsupported device option: {device_pref!r}")


class MnistMLP(Module[[Tensor], Tensor]):
    """3-layer MLP (`784 -> 512 -> 256 -> 10`) using MAX `Linear` modules."""

    def __init__(self) -> None:
        super().__init__()
        with default_dtype(DType.float32):
            self.fc1 = Linear(IMG_PIXELS, HIDDEN1, bias=True)
            self.fc2 = Linear(HIDDEN1, HIDDEN2, bias=True)
            self.fc3 = Linear(HIDDEN2, NUM_CLASSES, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MnistEvalWrapper(Module[[Tensor], tuple[Tensor, Tensor]]):
    """Compiled evaluation wrapper returning `(logits, predicted_classes)`."""

    def __init__(self, mlp: MnistMLP) -> None:
        super().__init__()
        self.mlp = mlp

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.mlp(x)
        preds = F.argmax(logits, axis=-1)
        return logits, preds


class EvalStep(Module):
    """Compiled stateless evaluation step so training never recompiles per epoch."""

    def forward(
        self,
        x: Tensor,
        w1: Tensor,
        b1: Tensor,
        w2: Tensor,
        b2: Tensor,
        w3: Tensor,
        b3: Tensor,
    ) -> tuple[Tensor, Tensor]:
        z1 = F.matmul(x, w1.T) + b1
        a1 = F.relu(z1)
        z2 = F.matmul(a1, w2.T) + b2
        a2 = F.relu(z2)
        logits = F.matmul(a2, w3.T) + b3
        preds = F.argmax(logits, axis=-1)
        return logits, preds


def tensor_scalar(t: Tensor) -> float:
    """Extracts a scalar float from a 0D/1D MAX Tensor."""
    return float(np.from_dlpack(t.to(CPU())).reshape(-1)[0])


class TrainStep(Module):
    """Single compiled MAX training step: forward, loss, backward, and AdamW."""

    def __init__(self, batch_size: int) -> None:
        super().__init__()
        self.batch_size = batch_size

    def _adamw_update(
        self,
        param: Tensor,
        grad: Tensor,
        m: Tensor,
        v: Tensor,
        lr: Tensor,
        inv_bc1: Tensor,
        inv_bc2: Tensor,
        wd: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        m_new = m * BETA1 + grad * (1.0 - BETA1)
        v_new = v * BETA2 + (grad * grad) * (1.0 - BETA2)
        m_hat = m_new * inv_bc1
        v_hat = v_new * inv_bc2
        eps = Tensor(ADAM_EPS, dtype=DType.float32, device=param.device)
        step = lr * m_hat / (F.sqrt(v_hat) + eps)
        if wd > 0.0:
            param_new = param * (1.0 - lr * wd) - step
        else:
            param_new = param - step
        return param_new, m_new, v_new

    def forward(
        self,
        x: Tensor,
        y_smooth: Tensor,
        y_idx: Tensor,
        drop1: Tensor,
        drop2: Tensor,
        lr: Tensor,
        inv_bc1: Tensor,
        inv_bc2: Tensor,
        w1: Tensor,
        b1: Tensor,
        w2: Tensor,
        b2: Tensor,
        w3: Tensor,
        b3: Tensor,
        m_w1: Tensor,
        m_b1: Tensor,
        m_w2: Tensor,
        m_b2: Tensor,
        m_w3: Tensor,
        m_b3: Tensor,
        v_w1: Tensor,
        v_b1: Tensor,
        v_w2: Tensor,
        v_b2: Tensor,
        v_w3: Tensor,
        v_b3: Tensor,
    ) -> tuple[Tensor, ...]:
        inv_batch = 1.0 / float(self.batch_size)
        eps_prob = Tensor(1e-7, dtype=DType.float32, device=x.device)

        # Forward pass with inverted dropout masks
        z1 = F.matmul(x, w1.T) + b1
        a1 = F.relu(z1) * drop1
        z2 = F.matmul(a1, w2.T) + b2
        a2 = F.relu(z2) * drop2
        logits = F.matmul(a2, w3.T) + b3

        # Softmax + label-smoothed cross-entropy + accuracy
        probs = F.softmax(logits, axis=-1)
        safe_probs = F.elementwise_max(probs, eps_prob)
        per_sample_loss = F.squeeze(F.sum(y_smooth * F.log(safe_probs), axis=-1), axis=-1)
        loss = -F.squeeze(F.sum(per_sample_loss, axis=0), axis=0) * inv_batch
        preds = F.squeeze(F.argmax(logits, axis=-1), axis=-1)
        correct = F.sum(F.cast(F.equal(preds, y_idx), DType.float32))

        # Backward pass
        dz3 = (probs - y_smooth) * inv_batch
        dw3 = F.matmul(dz3.T, a2)
        db3 = F.squeeze(F.sum(dz3, axis=0), axis=0)

        da2 = F.matmul(dz3, w3)
        dz2 = da2 * drop2 * F.cast(a2 > 0.0, DType.float32)
        dw2 = F.matmul(dz2.T, a1)
        db2 = F.squeeze(F.sum(dz2, axis=0), axis=0)

        da1 = F.matmul(dz2, w2)
        dz1 = da1 * drop1 * F.cast(a1 > 0.0, DType.float32)
        dw1 = F.matmul(dz1.T, x)
        db1 = F.squeeze(F.sum(dz1, axis=0), axis=0)

        # AdamW updates
        w1, m_w1, v_w1 = self._adamw_update(w1, dw1, m_w1, v_w1, lr, inv_bc1, inv_bc2, WEIGHT_DECAY)
        b1, m_b1, v_b1 = self._adamw_update(b1, db1, m_b1, v_b1, lr, inv_bc1, inv_bc2, 0.0)
        w2, m_w2, v_w2 = self._adamw_update(w2, dw2, m_w2, v_w2, lr, inv_bc1, inv_bc2, WEIGHT_DECAY)
        b2, m_b2, v_b2 = self._adamw_update(b2, db2, m_b2, v_b2, lr, inv_bc1, inv_bc2, 0.0)
        w3, m_w3, v_w3 = self._adamw_update(w3, dw3, m_w3, v_w3, lr, inv_bc1, inv_bc2, WEIGHT_DECAY)
        b3, m_b3, v_b3 = self._adamw_update(b3, db3, m_b3, v_b3, lr, inv_bc1, inv_bc2, 0.0)

        return (
            loss,
            correct,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            m_w1,
            m_b1,
            m_w2,
            m_b2,
            m_w3,
            m_b3,
            v_w1,
            v_b1,
            v_w2,
            v_b2,
            v_w3,
            v_b3,
        )


def init_he_state(device: Device, seed: int = 42) -> list[Tensor]:
    """Initializes weights with He normal initialization and zero AdamW state."""
    rng = np.random.default_rng(seed)

    def he_weight(out_dim: int, in_dim: int) -> Tensor:
        std = np.sqrt(2.0 / float(in_dim))
        arr = rng.normal(0.0, std, size=(out_dim, in_dim)).astype(np.float32)
        return Tensor.from_dlpack(arr).to(device)

    w1 = he_weight(HIDDEN1, IMG_PIXELS)
    b1 = Tensor.zeros([HIDDEN1], dtype=DType.float32, device=device)
    w2 = he_weight(HIDDEN2, HIDDEN1)
    b2 = Tensor.zeros([HIDDEN2], dtype=DType.float32, device=device)
    w3 = he_weight(NUM_CLASSES, HIDDEN2)
    b3 = Tensor.zeros([NUM_CLASSES], dtype=DType.float32, device=device)

    zeros_like = [
        Tensor.zeros([HIDDEN1, IMG_PIXELS], dtype=DType.float32, device=device),
        Tensor.zeros([HIDDEN1], dtype=DType.float32, device=device),
        Tensor.zeros([HIDDEN2, HIDDEN1], dtype=DType.float32, device=device),
        Tensor.zeros([HIDDEN2], dtype=DType.float32, device=device),
        Tensor.zeros([NUM_CLASSES, HIDDEN2], dtype=DType.float32, device=device),
        Tensor.zeros([NUM_CLASSES], dtype=DType.float32, device=device),
    ]
    zeros_like_v = [
        Tensor.zeros([HIDDEN1, IMG_PIXELS], dtype=DType.float32, device=device),
        Tensor.zeros([HIDDEN1], dtype=DType.float32, device=device),
        Tensor.zeros([HIDDEN2, HIDDEN1], dtype=DType.float32, device=device),
        Tensor.zeros([HIDDEN2], dtype=DType.float32, device=device),
        Tensor.zeros([NUM_CLASSES, HIDDEN2], dtype=DType.float32, device=device),
        Tensor.zeros([NUM_CLASSES], dtype=DType.float32, device=device),
    ]
    return [w1, b1, w2, b2, w3, b3, *zeros_like, *zeros_like_v]


def compile_train_step(batch_size: int, device: Device):
    """Compiles the fused MAX training step for the given batch size and device."""
    step = TrainStep(batch_size=batch_size).to(device)
    input_types = [
        TensorType(DType.float32, [batch_size, IMG_PIXELS], device),  # x
        TensorType(DType.float32, [batch_size, NUM_CLASSES], device),  # y_smooth
        TensorType(DType.int64, [batch_size], device),  # y_idx
        TensorType(DType.float32, [batch_size, HIDDEN1], device),  # drop1
        TensorType(DType.float32, [batch_size, HIDDEN2], device),  # drop2
        TensorType(DType.float32, [], device),  # lr
        TensorType(DType.float32, [], device),  # inv_bc1
        TensorType(DType.float32, [], device),  # inv_bc2
        TensorType(DType.float32, [HIDDEN1, IMG_PIXELS], device),  # w1
        TensorType(DType.float32, [HIDDEN1], device),  # b1
        TensorType(DType.float32, [HIDDEN2, HIDDEN1], device),  # w2
        TensorType(DType.float32, [HIDDEN2], device),  # b2
        TensorType(DType.float32, [NUM_CLASSES, HIDDEN2], device),  # w3
        TensorType(DType.float32, [NUM_CLASSES], device),  # b3
        TensorType(DType.float32, [HIDDEN1, IMG_PIXELS], device),  # m_w1
        TensorType(DType.float32, [HIDDEN1], device),  # m_b1
        TensorType(DType.float32, [HIDDEN2, HIDDEN1], device),  # m_w2
        TensorType(DType.float32, [HIDDEN2], device),  # m_b2
        TensorType(DType.float32, [NUM_CLASSES, HIDDEN2], device),  # m_w3
        TensorType(DType.float32, [NUM_CLASSES], device),  # m_b3
        TensorType(DType.float32, [HIDDEN1, IMG_PIXELS], device),  # v_w1
        TensorType(DType.float32, [HIDDEN1], device),  # v_b1
        TensorType(DType.float32, [HIDDEN2, HIDDEN1], device),  # v_w2
        TensorType(DType.float32, [HIDDEN2], device),  # v_b2
        TensorType(DType.float32, [NUM_CLASSES, HIDDEN2], device),  # v_w3
        TensorType(DType.float32, [NUM_CLASSES], device),  # v_b3
    ]
    return step.compile(*input_types)


def compile_eval_step(batch_size: int, device: Device):
    """Compiles a stateless MAX evaluation step for the given batch size and device."""
    step = EvalStep().to(device)
    return step.compile(
        TensorType(DType.float32, [batch_size, IMG_PIXELS], device),
        TensorType(DType.float32, [HIDDEN1, IMG_PIXELS], device),
        TensorType(DType.float32, [HIDDEN1], device),
        TensorType(DType.float32, [HIDDEN2, HIDDEN1], device),
        TensorType(DType.float32, [HIDDEN2], device),
        TensorType(DType.float32, [NUM_CLASSES, HIDDEN2], device),
        TensorType(DType.float32, [NUM_CLASSES], device),
    )


def save_weights_npz(path: str, params: list[Tensor]) -> None:
    """Saves trained parameters `(w1, b1, w2, b2, w3, b3)` to `.npz`."""
    names = ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias")
    arrays = {name: np.from_dlpack(p.to(CPU())).copy() for name, p in zip(names, params)}
    np.savez(path, **arrays)


def load_mlp_for_inference(weights_path: str, batch_size: int, device: Device):
    """Loads `.npz` weights into `MnistMLP` and compiles it on MAX."""
    mlp = MnistMLP()
    with np.load(weights_path) as data:
        state_dict = {k: data[k].astype(np.float32) for k in data.files}
    mlp.load_state_dict(state_dict)
    mlp = mlp.to(device)
    eval_wrapper = MnistEvalWrapper(mlp).to(device)
    return eval_wrapper.compile(TensorType(DType.float32, [batch_size, IMG_PIXELS], device))
