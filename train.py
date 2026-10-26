"""Handwritten Digit Recognition - model training.

OOP mini-project: the whole pipeline is modelled as small classes.

    MNISTDataset      -> loads and batches the MNIST digit images
    NeuralNetwork     -> a fully connected net (forward / backward / predict)
    Trainer           -> runs the training + evaluation loop
    ModelExporter     -> writes the trained weights to web/model.json

Run:  python3 train.py
"""

from __future__ import annotations

import gzip
import json
import os
import struct
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "model.json")

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class MNISTDataset:
    """Loads the MNIST IDX files and serves mini-batches."""

    IMAGE_SIZE = 28 * 28
    NUM_CLASSES = 10
    BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.ensure_downloaded()
        self.x_train, self.y_train = self._load("train_images", "train_labels")
        self.x_test, self.y_test = self._load("test_images", "test_labels")

    def ensure_downloaded(self) -> None:
        """Fetches any missing MNIST file (~11 MB in total)."""
        os.makedirs(self.data_dir, exist_ok=True)
        for filename in MNIST_FILES.values():
            path = os.path.join(self.data_dir, filename)
            if os.path.exists(path):
                continue
            print(f"downloading {filename} ...")
            urllib.request.urlretrieve(self.BASE_URL + filename, path)

    @staticmethod
    def _read_idx(path: str) -> np.ndarray:
        with gzip.open(path, "rb") as f:
            magic, = struct.unpack(">I", f.read(4))
            ndim = magic & 0xFF
            dims = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(dims)

    def _load(self, images_key: str, labels_key: str):
        images = self._read_idx(os.path.join(self.data_dir, MNIST_FILES[images_key]))
        labels = self._read_idx(os.path.join(self.data_dir, MNIST_FILES[labels_key]))
        # Flatten to vectors of 784 floats in [0, 1], same as the browser will do.
        x = images.reshape(-1, self.IMAGE_SIZE).astype(np.float32) / 255.0
        return x, labels.astype(np.int64)

    def batches(self, x: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator):
        """Yields shuffled mini-batches."""
        idx = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            sel = idx[start:start + batch_size]
            yield x[sel], y[sel]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class NeuralNetwork:
    """A fully connected network: 784 -> 128 -> 64 -> 10, ReLU hidden layers."""

    def __init__(self, layer_sizes: list[int], seed: int = 42):
        self.layer_sizes = layer_sizes
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = np.sqrt(2.0 / fan_in)  # He initialisation
            self.weights.append(rng.normal(0.0, limit, size=(fan_in, fan_out)).astype(np.float32))
            self.biases.append(np.zeros(fan_out, dtype=np.float32))
        self._adam_m = [np.zeros_like(w) for w in self.weights]
        self._adam_v = [np.zeros_like(w) for w in self.weights]
        self._adam_mb = [np.zeros_like(b) for b in self.biases]
        self._adam_vb = [np.zeros_like(b) for b in self.biases]
        self._adam_t = 0

    # -- forward ----------------------------------------------------------- #
    def forward(self, x: np.ndarray, cache: bool = False):
        """Returns the class probabilities; optionally caches activations."""
        activations = [x]
        z_values = []
        a = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            z_values.append(z)
            if i < len(self.weights) - 1:
                a = np.maximum(z, 0.0)          # ReLU
            else:
                a = self._softmax(z)            # output layer
            activations.append(a)
        if cache:
            return a, (activations, z_values)
        return a

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # -- backward ---------------------------------------------------------- #
    def backward(self, y: np.ndarray, cache) -> list[float]:
        """Computes gradients for a mini-batch of integer labels."""
        activations, z_values = cache
        n = y.shape[0]
        grads_w, grads_b = [], []

        delta = activations[-1].copy()
        delta[np.arange(n), y] -= 1.0            # softmax + cross-entropy
        delta /= n

        for i in range(len(self.weights) - 1, -1, -1):
            gw = activations[i].T @ delta
            gb = delta.sum(axis=0)
            grads_w.append(gw)
            grads_b.append(gb)
            if i > 0:
                delta = (delta @ self.weights[i].T) * (z_values[i - 1] > 0)  # ReLU derivative
        return grads_w[::-1], grads_b[::-1]

    def apply_gradients(self, grads_w, grads_b, lr: float) -> None:
        """One Adam update step."""
        self._adam_t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for i in range(len(self.weights)):
            self._adam_m[i] = beta1 * self._adam_m[i] + (1 - beta1) * grads_w[i]
            self._adam_v[i] = beta2 * self._adam_v[i] + (1 - beta2) * grads_w[i] ** 2
            self._adam_mb[i] = beta1 * self._adam_mb[i] + (1 - beta1) * grads_b[i]
            self._adam_vb[i] = beta2 * self._adam_vb[i] + (1 - beta2) * grads_b[i] ** 2
            m_hat = self._adam_m[i] / (1 - beta1 ** self._adam_t)
            v_hat = self._adam_v[i] / (1 - beta2 ** self._adam_t)
            mb_hat = self._adam_mb[i] / (1 - beta1 ** self._adam_t)
            vb_hat = self._adam_vb[i] / (1 - beta2 ** self._adam_t)
            self.weights[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
            self.biases[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    # -- evaluation -------------------------------------------------------- #
    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x).argmax(axis=1)

    def accuracy(self, x: np.ndarray, y: np.ndarray, batch_size: int = 1000) -> float:
        correct = 0
        for start in range(0, len(x), batch_size):
            chunk = x[start:start + batch_size]
            correct += (self.predict(chunk) == y[start:start + batch_size]).sum()
        return correct / len(x)


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
class Trainer:
    def __init__(self, model: NeuralNetwork, dataset: MNISTDataset,
                 epochs: int = 8, batch_size: int = 128, lr: float = 1e-3, seed: int = 0):
        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
        return -np.log(probs[np.arange(len(y)), y] + 1e-9).mean()

    def run(self) -> dict:
        history = []
        for epoch in range(1, self.epochs + 1):
            start = time.time()
            losses = []
            for xb, yb in self.dataset.batches(
                self.dataset.x_train, self.dataset.y_train, self.batch_size, self.rng
            ):
                probs, cache = self.model.forward(xb, cache=True)
                losses.append(self.cross_entropy(probs, yb))
                grads_w, grads_b = self.model.backward(yb, cache)
                self.model.apply_gradients(grads_w, grads_b, self.lr)
            test_acc = self.model.accuracy(self.dataset.x_test, self.dataset.y_test)
            stats = {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "test_accuracy": float(test_acc),
                "seconds": round(time.time() - start, 1),
            }
            history.append(stats)
            print(f"epoch {epoch:>2}/{self.epochs}  loss {stats['loss']:.4f}  "
                  f"test acc {test_acc * 100:.2f}%  ({stats['seconds']}s)")
        return {"history": history, "test_accuracy": history[-1]["test_accuracy"]}


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
class ModelExporter:
    """Writes the trained weights as JSON that the browser app can consume."""

    def __init__(self, model: NeuralNetwork, precision: int = 5):
        self.model = model
        self.precision = precision

    def _round(self, arr: np.ndarray) -> list:
        rounded = np.round(arr.astype(np.float64), self.precision)
        return rounded.tolist()

    def export(self, path: str, metrics: dict) -> str:
        payload = {
            "format": "digit-recognizer-mlp-v1",
            "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "layerSizes": self.model.layer_sizes,
            "testAccuracy": round(metrics["test_accuracy"], 4),
            "history": metrics["history"],
            "layers": [
                {"W": self._round(w), "b": self._round(b)}
                for w, b in zip(self.model.weights, self.model.biases)
            ],
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        return path


# --------------------------------------------------------------------------- #
def main() -> None:
    print("loading MNIST ...")
    dataset = MNISTDataset()
    print(f"  train {dataset.x_train.shape}, test {dataset.x_test.shape}")

    model = NeuralNetwork([784, 128, 64, 10])
    trainer = Trainer(model, dataset, epochs=8, batch_size=128, lr=1e-3)
    metrics = trainer.run()

    path = ModelExporter(model).export(MODEL_PATH, metrics)
    print(f"\ntest accuracy: {metrics['test_accuracy'] * 100:.2f}%")
    print(f"model written to {path} ({os.path.getsize(path) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
