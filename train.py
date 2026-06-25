"""
train.py — headless end-to-end training pipeline.

Runs the full classification pipeline in one process (ideal for a GPU VM):
    load corpus -> clean (route by language) -> stratified split
    -> TF-IDF baseline (floor) -> DistilBERT fine-tuning (TensorFlow/Keras)
    -> evaluation, per-language accuracy, and report artifacts.

This is the *training* counterpart to the EDA notebook; it reuses the same OOP
components (utils.text_cleaning, models.text_classifier). Translation is NOT done
here — it expects a pre-built corpus at data/processed_data/multilingual_corpus.csv
(see utils/build_multilingual_dataset.py).

Usage:
    python train.py                 # 5 epochs on the existing corpus
    python train.py --epochs 5 --corpus data/processed_data/multilingual_corpus.csv

Outputs:
    models/checkpoints/{text_classifier_best.h5, config.json, training_history.csv}
    reports/performance_metrics.json
    reports/example_predictions.csv
"""
import os

# Keep transformers TensorFlow-only (the classifier is TF/Keras; avoids importing
# PyTorch alongside TF). Must be set before transformers is imported.
os.environ.setdefault("USE_TORCH", "0")

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from utils.text_cleaning import LanguageSpecificSanitizer
from models.text_classifier import BaselineClassifier, DistilBertClassifier

DEFAULT_CORPUS = "data/processed_data/multilingual_corpus.csv"
REPORTS_DIR = "reports"
MIN_WORDS = 3


def load_clean_split(
    corpus_path: str, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the multi-language corpus, cleans each document by its known language,
    drops empties/micro docs, and returns a stratified train/validation split.

    Args:
        corpus_path (str): Path to the multilingual corpus CSV.
        test_size (float): Validation fraction.
        seed (int): Random seed for the split.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_df, val_df), each with a
        ``clean_text`` column plus the original ``label`` / ``language``.
    """
    try:
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(
                f"{corpus_path} not found. Build it with "
                "`python -m utils.build_multilingual_dataset` first."
            )
        dataframe = pd.read_csv(corpus_path)
        logger.info(
            f"Loaded {len(dataframe)} docs: {dataframe['language'].value_counts().to_dict()}"
        )

        sanitizers = {
            lang: LanguageSpecificSanitizer(lang) for lang in ("en", "sv", "fi")
        }

        def clean_one(row: pd.Series) -> str:
            text = "" if pd.isna(row["text"]) else str(row["text"])
            if not text.strip():
                return None
            cleaned = sanitizers.get(row["language"], sanitizers["en"]).clean(text)
            return cleaned if len(cleaned.split()) >= MIN_WORDS else None

        dataframe["clean_text"] = dataframe.apply(clean_one, axis=1)
        dataframe = dataframe.dropna(subset=["clean_text"]).reset_index(drop=True)
        logger.info(f"{len(dataframe)} docs remain after cleaning.")

        train_df, val_df = train_test_split(
            dataframe,
            test_size=test_size,
            random_state=seed,
            stratify=dataframe["label"],
        )
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
    except Exception as e:
        logger.error(f"load_clean_split failed: {e}")
        raise


def per_language_accuracy(
    model: DistilBertClassifier, val_df: pd.DataFrame
) -> dict[str, float]:
    """
    Computes validation accuracy separately for each language.

    Args:
        model (DistilBertClassifier): A trained classifier returning probabilities.
        val_df (pd.DataFrame): Validation split with ``clean_text``/``label``/``language``.

    Returns:
        dict[str, float]: Per-language accuracy (e.g. {"en": 0.88, "sv": 0.84}).
    """
    try:
        results: dict[str, float] = {}
        for lang, group in val_df.groupby("language"):
            probs = model.predict(group["clean_text"].tolist())
            y_pred = np.argmax(probs, axis=1)
            results[lang] = round(float((y_pred == group["label"].values).mean()), 4)
        return results
    except Exception as e:
        logger.error(f"per_language_accuracy failed: {e}")
        raise


def main(epochs: int, corpus_path: str) -> None:
    """Runs the full baseline + DistilBERT training pipeline and writes artifacts."""
    try:
        # Show INFO and above; the cleaner emits one DEBUG line per document.
        logger.remove()
        logger.add(sys.stderr, level="INFO")

        os.makedirs(REPORTS_DIR, exist_ok=True)
        train_df, val_df = load_clean_split(corpus_path)
        X_train, y_train = train_df["clean_text"].tolist(), train_df["label"].tolist()
        X_val, y_val = val_df["clean_text"].tolist(), val_df["label"].tolist()
        num_classes = len(set(y_train))
        logger.info(f"Train {len(X_train)} | Val {len(X_val)} | classes {num_classes}")

        # 1. Baseline (the floor DistilBERT must beat by >= 5%).
        logger.info("=== Baseline (TF-IDF + Logistic Regression) ===")
        baseline = BaselineClassifier()
        baseline.train(X_train, y_train)
        base_metrics = baseline.evaluate(X_val, y_val)

        # 2. DistilBERT fine-tuning (saves checkpoints to models/checkpoints/).
        logger.info("=== DistilBERT fine-tuning ===")
        deep_model = DistilBertClassifier(num_classes=num_classes)
        deep_model.train(X_train, y_train, X_val, y_val, epochs=epochs)
        dl_metrics = deep_model.evaluate(X_val, y_val)

        # 3. Throughput (docs/sec) over the validation set.
        start = time.time()
        deep_model.predict(X_val)
        docs_per_sec = round(len(X_val) / max(1e-9, time.time() - start), 1)

        # 4. Assemble the performance report (audit schema).
        report = {
            "classification_accuracy": round(float(dl_metrics["accuracy"]), 4),
            "f1_score_macro": round(float(dl_metrics["f1_macro"]), 4),
            "processing_speed_docs_per_sec": docs_per_sec,
            "languages_supported": sorted(val_df["language"].unique().tolist()),
            "per_language_accuracy": per_language_accuracy(deep_model, val_df),
            "baseline_accuracy": round(float(base_metrics["accuracy"]), 4),
            "baseline_f1_macro": round(float(base_metrics["f1_macro"]), 4),
        }
        with open(os.path.join(REPORTS_DIR, "performance_metrics.json"), "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Wrote reports/performance_metrics.json: {report}")

        # 5. Sample predictions for inspection.
        sample = val_df.sample(min(20, len(val_df)), random_state=42).copy()
        sample["predicted_label"] = np.argmax(
            deep_model.predict(sample["clean_text"].tolist()), axis=1
        )
        sample[["text", "label", "predicted_label", "language"]].to_csv(
            os.path.join(REPORTS_DIR, "example_predictions.csv"), index=False
        )
        logger.info("Wrote reports/example_predictions.csv. Training pipeline complete.")
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the document classifier.")
    parser.add_argument("--epochs", type=int, default=5, help="Fine-tuning epochs.")
    parser.add_argument("--corpus", type=str, default=DEFAULT_CORPUS, help="Corpus CSV path.")
    args = parser.parse_args()
    main(epochs=args.epochs, corpus_path=args.corpus)
