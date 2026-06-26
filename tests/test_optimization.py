import os

os.environ.setdefault("USE_TORCH", "0")

import numpy as np
import pytest
import tensorflow as tf

from models.optimization import (
    MagnitudePruningOptimizer,
    Float16Quantizer,
    TFLiteDynamicRangeQuantizer,
    _file_size_mb,
)


def _tiny_model() -> tf.keras.Model:
    """A small Keras model with rank-2 Dense kernels (prunable) and 1-D biases."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(8),
            tf.keras.layers.Dense(4),
        ]
    )
    return model


def test_file_size_mb_missing_returns_zero() -> None:
    """A non-existent path reports 0.0 MB rather than raising."""
    assert _file_size_mb("/no/such/file.bin") == 0.0


def test_prune_tensor_zeros_smallest_magnitudes(tmp_path) -> None:
    """Pruning zeros exactly the smallest-magnitude weights (50% here)."""
    opt = MagnitudePruningOptimizer(
        _tiny_model(), str(tmp_path / "p.weights.h5"), target_sparsity=0.5
    )
    pruned = opt._prune_tensor(np.array([[-1.0, 2.0, -3.0, 4.0]]))
    # Two smallest by |.| (1 and 2) -> zero; 3 and 4 kept.
    assert pruned.tolist() == [[0.0, 0.0, -3.0, 4.0]]


def test_magnitude_pruning_reaches_target_sparsity(tmp_path) -> None:
    """The pruned model hits ~target sparsity and writes an artifact."""
    out = str(tmp_path / "pruned.weights.h5")
    metrics = MagnitudePruningOptimizer(
        _tiny_model(), out, target_sparsity=0.5
    ).optimize()
    assert metrics["technique"] == "magnitude_pruning"
    assert 0.4 <= metrics["achieved_sparsity"] <= 0.6
    assert metrics["tensors_pruned"] == 2  # two Dense kernels (biases skipped)
    assert os.path.exists(out)


def test_pruning_rejects_invalid_sparsity(tmp_path) -> None:
    """A sparsity outside [0, 1) raises a controlled ValueError."""
    with pytest.raises(ValueError):
        MagnitudePruningOptimizer(
            _tiny_model(), str(tmp_path / "x.h5"), target_sparsity=1.0
        )


def test_float16_quantizer_saves_and_rounds(tmp_path) -> None:
    """fp16 quantization writes a compressed artifact and rounds the live weights."""
    model = _tiny_model()
    out = str(tmp_path / "fp16.npz")
    metrics = Float16Quantizer(model, out).optimize()
    assert metrics["technique"] == "float16_quantization"
    assert os.path.exists(out)
    # Every live weight now equals its float16-rounded value.
    for w in model.get_weights():
        assert np.array_equal(w, w.astype(np.float16).astype(np.float32))


def test_tflite_quantizer_degrades_gracefully(tmp_path) -> None:
    """A non-transformer model can't convert with the HF signature -> status 'failed',
    captured rather than raised so the wider run still completes."""
    result = TFLiteDynamicRangeQuantizer(
        _tiny_model(), str(tmp_path / "m.tflite"), max_length=16
    ).optimize()
    assert result["technique"] == "tflite_dynamic_range_int8"
    assert result["status"] in {"ok", "failed"}
