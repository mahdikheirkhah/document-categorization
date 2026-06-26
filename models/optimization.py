"""
models/optimization.py

Model-optimization layer (issue #5): shrink and speed up the fine-tuned DistilBERT
classifier through **pruning** and **quantization**, while measuring how much
accuracy each technique costs.

We implement the techniques with **core TensorFlow** rather than the
``tensorflow_model_optimization`` toolkit: tfmot's pruning needs prune-aware
re-training and does not cleanly wrap Hugging Face's custom transformer layers,
whereas magnitude pruning + post-training quantization are robust, dependency-free,
and run without a GPU.

OOP design:
    * Abstraction   -> BaseModelOptimizer defines the ``optimize`` contract.
    * Inheritance   -> MagnitudePruningOptimizer / Float16Quantizer /
                       TFLiteDynamicRangeQuantizer extend it.
    * Encapsulation -> each optimizer owns its artifact path and private helpers.
    * Polymorphism  -> ModelOptimizationPipeline drives a list of BaseModelOptimizer.

Techniques:
    * Magnitude pruning  -> zero the smallest-magnitude weights in the large 2-D
      kernels/embeddings. Pruned weights reload into the SAME architecture, so the
      serving pipeline needs no new code; the zeros also make the model far more
      compressible.
    * Float16 quantization -> halve weight precision (32->16 bit): ~2x smaller.
    * Dynamic-range int8 (TFLite) -> int8 weights, float activations: ~4x smaller
      and faster CPU inference.
"""

import json
import os
import time
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from loguru import logger

from utils import config


def _file_size_mb(path: str) -> float:
    """Returns a file's size in megabytes (0.0 if it does not exist)."""
    try:
        return round(os.path.getsize(path) / 1e6, 2) if os.path.exists(path) else 0.0
    except OSError as e:
        logger.error(f"Could not stat '{path}': {e}")
        return 0.0


class BaseModelOptimizer(ABC):
    """
    Abstract base for every model-optimization technique.

    Subclasses transform a Keras model (pruning, quantization, ...) and persist an
    artifact, returning a metrics dict describing the result.
    """

    def __init__(self, model: tf.keras.Model, output_path: str) -> None:
        """
        Args:
            model (tf.keras.Model): The model to optimize (mutated in place for
                pruning; read-only for the quantizers).
            output_path (str): Where the optimized artifact is written.
        """
        try:
            self.model = model
            self.output_path = output_path
            logger.info(f"Initialized {type(self).__name__} -> {output_path}")
        except Exception as e:
            logger.error(f"Failed to initialize {type(self).__name__}: {e}")
            raise

    @abstractmethod
    def optimize(self) -> dict:
        """
        Runs the optimization, writes the artifact, and returns its metrics.

        Returns:
            dict: Technique-specific metrics (always includes ``technique`` and
            ``artifact``).
        """
        pass


class MagnitudePruningOptimizer(BaseModelOptimizer):
    """
    Post-training magnitude pruning: for every weight tensor of rank >= ``min_rank``
    (the big kernels/embeddings, not 1-D biases/LayerNorm), zero the
    ``target_sparsity`` fraction of smallest-magnitude weights **in place**, then
    save the sparse weights.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        output_path: str,
        target_sparsity: float = config.PRUNING_TARGET_SPARSITY,
        min_rank: int = config.PRUNING_MIN_RANK,
    ) -> None:
        """
        Args:
            model (tf.keras.Model): Trained model to prune (mutated in place).
            output_path (str): Path for the pruned ``.weights.h5`` artifact.
            target_sparsity (float): Fraction of weights to zero per eligible tensor.
            min_rank (int): Minimum tensor rank to prune (skip 1-D biases/LayerNorm).
        """
        try:
            super().__init__(model, output_path)
            if not 0.0 <= target_sparsity < 1.0:
                raise ValueError("target_sparsity must be in [0.0, 1.0).")
            self.target_sparsity = target_sparsity
            self.min_rank = min_rank
        except Exception as e:
            logger.error(f"Failed to initialize MagnitudePruningOptimizer: {e}")
            raise

    def _prune_tensor(self, array: np.ndarray) -> np.ndarray:
        """
        Zeros the smallest-magnitude ``target_sparsity`` fraction of ``array``.

        Args:
            array (np.ndarray): The weight tensor values.

        Returns:
            np.ndarray: The pruned tensor (same shape, some entries set to 0).
        """
        try:
            k = int(self.target_sparsity * array.size)
            if k <= 0:
                return array
            # k-th smallest absolute value is the pruning threshold.
            threshold = np.partition(np.abs(array).ravel(), k - 1)[k - 1]
            return np.where(np.abs(array) > threshold, array, 0.0).astype(array.dtype)
        except Exception as e:
            logger.error(f"Failed to prune a weight tensor: {e}")
            raise

    def optimize(self) -> dict:
        """
        Prunes the model in place, saves the sparse weights, and reports sparsity.

        Returns:
            dict: {technique, target_sparsity, achieved_sparsity, params_total,
            params_zeroed, tensors_pruned, artifact, size_mb}.
        """
        try:
            pruned_elems = 0
            total_elems = 0
            tensors_pruned = 0
            for variable in self.model.weights:
                if variable.shape.rank is None or variable.shape.rank < self.min_rank:
                    continue
                values = variable.numpy()
                pruned = self._prune_tensor(values)
                variable.assign(pruned)
                total_elems += pruned.size
                pruned_elems += int((pruned == 0).sum())
                tensors_pruned += 1

            achieved = round(pruned_elems / max(1, total_elems), 4)
            self.model.save_weights(self.output_path)
            metrics = {
                "technique": "magnitude_pruning",
                "target_sparsity": self.target_sparsity,
                "achieved_sparsity": achieved,
                "params_total_pruned_tensors": int(total_elems),
                "params_zeroed": int(pruned_elems),
                "tensors_pruned": tensors_pruned,
                "artifact": os.path.basename(self.output_path),
                "size_mb": _file_size_mb(self.output_path),
            }
            logger.info(
                f"Pruned {tensors_pruned} tensors to {achieved:.1%} sparsity "
                f"({pruned_elems:,} weights zeroed)."
            )
            return metrics
        except Exception as e:
            logger.error(f"Magnitude pruning failed: {e}")
            raise


class Float16Quantizer(BaseModelOptimizer):
    """
    Post-training float16 quantization: store weights at half precision (32->16
    bit), roughly halving on-disk size. Optionally rounds the live model's weights
    to float16 precision so accuracy can be measured under quantization.
    """

    def optimize(self) -> dict:
        """
        Saves float16 weights (compressed ``.npz``) and rounds the live model to
        float16 precision so its post-quantization accuracy can be evaluated.

        Returns:
            dict: {technique, artifact, size_mb, params}.
        """
        try:
            weights = self.model.get_weights()
            fp16 = [w.astype(np.float16) for w in weights]
            np.savez_compressed(self.output_path, *fp16)

            # Apply fp16 rounding to the live model so a later eval reflects the
            # precision loss (cast back to fp32 — the storage stays fp32 but the
            # VALUES are the fp16-rounded ones).
            self.model.set_weights([w.astype(np.float32) for w in fp16])

            metrics = {
                "technique": "float16_quantization",
                "artifact": os.path.basename(self.output_path),
                "size_mb": _file_size_mb(self.output_path),
                "params": int(sum(w.size for w in weights)),
            }
            logger.info(f"Saved float16 weights ({metrics['size_mb']} MB).")
            return metrics
        except Exception as e:
            logger.error(f"Float16 quantization failed: {e}")
            raise


class TFLiteDynamicRangeQuantizer(BaseModelOptimizer):
    """
    Dynamic-range int8 quantization via TensorFlow Lite: weights become int8 while
    activations stay float, giving ~4x smaller models and faster CPU inference.

    Transformers use ops outside the core TFLite set, so the SELECT_TF_OPS (Flex)
    fallback is enabled. Conversion can still fail on some platforms; failures are
    captured in the returned metrics rather than raised, so the wider optimization
    run still completes.
    """

    def __init__(
        self, model: tf.keras.Model, output_path: str, max_length: int
    ) -> None:
        """
        Args:
            model (tf.keras.Model): The (Hugging Face) model to convert.
            output_path (str): Path for the ``.tflite`` artifact.
            max_length (int): Fixed sequence length for the serving signature.
        """
        try:
            super().__init__(model, output_path)
            self.max_length = max_length
        except Exception as e:
            logger.error(f"Failed to initialize TFLiteDynamicRangeQuantizer: {e}")
            raise

    def _concrete_function(self):
        """Builds a fixed-shape serving function (input_ids + attention_mask)."""
        signature = [
            {
                "input_ids": tf.TensorSpec(
                    [1, self.max_length], tf.int32, name="input_ids"
                ),
                "attention_mask": tf.TensorSpec(
                    [1, self.max_length], tf.int32, name="attention_mask"
                ),
            }
        ]

        @tf.function(input_signature=signature)
        def serving(inputs):
            return self.model(inputs, training=False).logits

        return serving.get_concrete_function()

    def optimize(self) -> dict:
        """
        Converts the model to dynamic-range int8 TFLite and writes the artifact.

        Returns:
            dict: {technique, status, artifact, size_mb[, error]}.
        """
        try:
            converter = tf.lite.TFLiteConverter.from_concrete_functions(
                [self._concrete_function()], self.model
            )
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # Transformers need the Flex delegate for unsupported ops.
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            tflite_model = converter.convert()
            with open(self.output_path, "wb") as f:
                f.write(tflite_model)
            metrics = {
                "technique": "tflite_dynamic_range_int8",
                "status": "ok",
                "artifact": os.path.basename(self.output_path),
                "size_mb": _file_size_mb(self.output_path),
            }
            logger.info(f"Saved int8 TFLite model ({metrics['size_mb']} MB).")
            return metrics
        except Exception as e:
            # Conversion of transformers is brittle across platforms; degrade
            # gracefully so pruning + fp16 results are still reported.
            logger.warning(f"TFLite int8 conversion unavailable on this platform: {e}")
            return {
                "technique": "tflite_dynamic_range_int8",
                "status": "failed",
                "artifact": None,
                "size_mb": 0.0,
                "error": str(e)[:300],
            }


class ModelOptimizationPipeline:
    """
    Orchestrates pruning + quantization on a trained classifier and measures the
    accuracy/size/speed of each stage, writing a consolidated JSON report.

    The optimizers are applied **cumulatively** to one model (prune -> float16 ->
    int8), mirroring a real deployment pipeline, and the model is re-measured after
    each step so the cost of every technique is visible in isolation.
    """

    def __init__(
        self,
        classifier,
        eval_texts: list[str],
        eval_labels: list[int],
        checkpoint_dir: str = None,
        report_path: str = config.OPTIMIZATION_METRICS_PATH,
    ) -> None:
        """
        Args:
            classifier: A trained ``DistilBertClassifier`` (duck-typed: needs
                ``model``, ``max_length`` and ``predict``).
            eval_texts (list[str]): Cleaned validation documents for measurement.
            eval_labels (list[int]): Their integer labels.
            checkpoint_dir (str, optional): Where artifacts are written. Defaults to
                ``config.CHECKPOINT_DIR``.
            report_path (str): Where the optimization report JSON is written.
        """
        try:
            self.classifier = classifier
            self.eval_texts = eval_texts
            self.eval_labels = np.asarray(eval_labels)
            self.checkpoint_dir = checkpoint_dir or config.CHECKPOINT_DIR
            self.report_path = report_path
            logger.info(
                f"Initialized ModelOptimizationPipeline on {len(eval_texts)} eval docs."
            )
        except Exception as e:
            logger.error(f"Failed to initialize ModelOptimizationPipeline: {e}")
            raise

    def _evaluate(self) -> tuple[float, float]:
        """
        Measures accuracy and throughput of the classifier's CURRENT weights.

        Returns:
            tuple[float, float]: (accuracy, docs_per_sec) over the eval set.
        """
        try:
            start = time.time()
            probabilities = self.classifier.predict(self.eval_texts)
            elapsed = max(1e-9, time.time() - start)
            predictions = np.argmax(probabilities, axis=1)
            accuracy = round(float((predictions == self.eval_labels).mean()), 4)
            docs_per_sec = round(len(self.eval_texts) / elapsed, 1)
            return accuracy, docs_per_sec
        except Exception as e:
            logger.error(f"Optimization eval failed: {e}")
            raise

    def run(self) -> dict:
        """
        Runs the full optimization sweep and writes the report.

        Returns:
            dict: The optimization report (also persisted to ``report_path``).
        """
        try:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
            model = self.classifier.model

            original_path = os.path.join(
                self.checkpoint_dir, config.BEST_WEIGHTS_FILENAME
            )
            base_acc, base_speed = self._evaluate()
            report = {
                "original": {
                    "size_mb": _file_size_mb(original_path),
                    "accuracy": base_acc,
                    "docs_per_sec": base_speed,
                }
            }
            logger.info(f"Original: acc={base_acc}, {base_speed} docs/s.")

            # 1. Magnitude pruning (mutates the model in place).
            pruned_path = os.path.join(
                self.checkpoint_dir, config.PRUNED_WEIGHTS_FILENAME
            )
            prune_metrics = MagnitudePruningOptimizer(model, pruned_path).optimize()
            acc, speed = self._evaluate()
            prune_metrics["accuracy"] = acc
            prune_metrics["accuracy_drop_vs_original"] = round(base_acc - acc, 4)
            prune_metrics["docs_per_sec"] = speed
            report["pruning"] = prune_metrics

            # 2. Float16 quantization (rounds the pruned model to fp16 precision).
            fp16_path = os.path.join(self.checkpoint_dir, config.FP16_WEIGHTS_FILENAME)
            fp16_metrics = Float16Quantizer(model, fp16_path).optimize()
            acc, speed = self._evaluate()
            fp16_metrics["accuracy"] = acc
            fp16_metrics["accuracy_drop_vs_original"] = round(base_acc - acc, 4)
            if report["original"]["size_mb"]:
                fp16_metrics["size_reduction_vs_original_pct"] = round(
                    100 * (1 - fp16_metrics["size_mb"] / report["original"]["size_mb"]),
                    1,
                )
            report["quantization_fp16"] = fp16_metrics

            # 3. Dynamic-range int8 via TFLite (best-effort; size + status).
            tflite_path = os.path.join(self.checkpoint_dir, config.TFLITE_FILENAME)
            tflite_metrics = TFLiteDynamicRangeQuantizer(
                model, tflite_path, self.classifier.max_length
            ).optimize()
            if tflite_metrics.get("size_mb") and report["original"]["size_mb"]:
                tflite_metrics["size_reduction_vs_original_pct"] = round(
                    100
                    * (1 - tflite_metrics["size_mb"] / report["original"]["size_mb"]),
                    1,
                )
            report["quantization_int8_tflite"] = tflite_metrics

            report["summary"] = {
                "eval_documents": len(self.eval_texts),
                "original_size_mb": report["original"]["size_mb"],
                "pruning_sparsity": prune_metrics["achieved_sparsity"],
                "fp16_size_mb": fp16_metrics["size_mb"],
                "int8_tflite_size_mb": tflite_metrics.get("size_mb", 0.0),
                "accuracy_original": base_acc,
                "accuracy_after_pruning": report["pruning"]["accuracy"],
                "accuracy_after_fp16": report["quantization_fp16"]["accuracy"],
            }

            with open(self.report_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Wrote optimization report to {self.report_path}.")
            return report
        except Exception as e:
            logger.error(f"ModelOptimizationPipeline.run failed: {e}")
            raise
