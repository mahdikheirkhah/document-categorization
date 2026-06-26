"""
optimize.py — headless model-optimization pipeline (issue #5).

Loads the fine-tuned DistilBERT checkpoint, applies pruning + quantization, and
writes the optimized artifacts plus a measurement report:

    models/checkpoints/text_classifier_pruned.weights.h5   (magnitude pruning)
    models/checkpoints/text_classifier_fp16.npz            (float16 quantization)
    models/checkpoints/text_classifier_int8.tflite         (dynamic-range int8)
    reports/optimization_metrics.json                      (size / accuracy / speed)

Usage:
    python optimize.py
    python optimize.py --eval-sample 400 --corpus data/processed_data/multilingual_corpus.csv
"""

import os

# Keep transformers TensorFlow-only (matches the training/inference path).
os.environ.setdefault("USE_TORCH", "0")

import argparse
import sys

from loguru import logger

from utils import config
from models.text_classifier import DistilBertClassifier
from models.optimization import ModelOptimizationPipeline
from train import load_clean_split


def main(eval_sample: int, corpus_path: str) -> None:
    """
    Runs pruning + quantization on the trained checkpoint and writes the report.

    Args:
        eval_sample (int): Number of validation documents used to measure each
            optimized variant's accuracy and speed.
        corpus_path (str): Path to the multilingual corpus CSV.
    """
    try:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

        # Reuse the exact training-time clean/split so eval matches validation.
        _, val_df = load_clean_split(corpus_path)
        sample = val_df.sample(min(eval_sample, len(val_df)), random_state=config.SEED)
        eval_texts = sample["clean_text"].tolist()
        eval_labels = sample["label"].tolist()
        logger.info(f"Evaluating optimization on {len(eval_texts)} validation docs.")

        classifier = DistilBertClassifier.from_checkpoint()
        report = ModelOptimizationPipeline(classifier, eval_texts, eval_labels).run()

        logger.info(f"Optimization complete. Summary: {report['summary']}")
    except Exception as e:
        logger.error(f"Optimization pipeline failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune + quantize the trained model.")
    parser.add_argument(
        "--eval-sample",
        type=int,
        default=config.OPTIMIZATION_EVAL_SAMPLE,
        help="Validation documents used to measure each optimized variant.",
    )
    parser.add_argument(
        "--corpus", type=str, default=config.CORPUS_PATH, help="Corpus CSV path."
    )
    args = parser.parse_args()
    main(eval_sample=args.eval_sample, corpus_path=args.corpus)
