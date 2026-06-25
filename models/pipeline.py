"""
models/pipeline.py

Real-time inference pipeline (issue #5). For each document it runs:
    language detection -> language-specific cleaning -> classification (DistilBERT)
    -> context-aware NER tagging,
returning one structured result. Classification is **batched** (a single model
call per batch) for throughput, and a ``benchmark`` helper measures docs/sec.

It reuses the trained checkpoint and the exact same OOP components as training
(``LanguageSpecificSanitizer``, ``DistilBertClassifier``, ``MultilingualTagger``,
``NgramLanguageDetector``), so inference matches training. Heavy dependencies are
injectable, which keeps the orchestration unit-testable without loading the model.
"""
import os

# Keep transformers TensorFlow-only (the classifier is TF/Keras).
os.environ.setdefault("USE_TORCH", "0")

import time

import numpy as np
import pandas as pd
from loguru import logger

from utils import config
from utils.language_detector import NgramLanguageDetector
from utils.text_cleaning import LanguageSpecificSanitizer
from models.text_classifier import DistilBertClassifier, CHECKPOINT_DIR
from models.tagger import MultilingualTagger

DEFAULT_CORPUS = config.CORPUS_PATH


class RealTimePipeline:
    """
    End-to-end document categorization + tagging for real-time use.

    Args:
        checkpoint_dir (str): Directory with the trained classifier checkpoint.
        corpus_path (str): Corpus used only to recover the {label -> category name} map.
        languages (list[str], optional): Supported languages. Defaults to en/sv/fi.
        classifier, tagger, detector: Optional pre-built components (dependency
            injection for testing). Built from defaults when omitted.
        label_map (dict[int, str], optional): Override the label->name mapping.
    """

    def __init__(
        self,
        checkpoint_dir: str = CHECKPOINT_DIR,
        corpus_path: str = DEFAULT_CORPUS,
        languages: list[str] = None,
        classifier=None,
        tagger=None,
        detector: NgramLanguageDetector = None,
        label_map: dict[int, str] = None,
    ) -> None:
        try:
            self.languages = languages or config.SUPPORTED_LANGUAGES
            self.detector = detector or NgramLanguageDetector(self.languages)
            self.sanitizers = {
                lang: LanguageSpecificSanitizer(lang) for lang in self.languages
            }
            self.tagger = tagger or MultilingualTagger(self.languages, detector=self.detector)
            self.classifier = (
                classifier
                if classifier is not None
                else DistilBertClassifier.from_checkpoint(checkpoint_dir)
            )
            self.label_map = (
                label_map if label_map is not None else self._load_label_map(corpus_path)
            )
            logger.info("Initialized RealTimePipeline.")
        except Exception as e:
            logger.error(f"Failed to initialize RealTimePipeline: {e}")
            raise

    def _load_label_map(self, corpus_path: str) -> dict[int, str]:
        """Builds {label_id: category_name} from the corpus; numeric fallback if absent."""
        try:
            if corpus_path and os.path.exists(corpus_path):
                frame = pd.read_csv(corpus_path, usecols=["label", "label_text"]).drop_duplicates()
                return {int(row.label): str(row.label_text) for row in frame.itertuples(index=False)}
            logger.warning(f"Corpus '{corpus_path}' not found; categories will be numeric.")
            return {}
        except Exception as e:
            logger.error(f"Failed to build label map: {e}")
            return {}

    def _detect(self, text: str) -> str:
        """Detects language, falling back to English on undetectable input."""
        try:
            return self.detector.detect_language(text)
        except Exception:
            return config.DEFAULT_LANGUAGE

    def _clean(self, text: str, language: str) -> str:
        """Cleans a document with the language-specific sanitizer."""
        sanitizer = self.sanitizers.get(language, self.sanitizers[config.DEFAULT_LANGUAGE])
        return sanitizer.clean(str(text))

    def process_batch(self, texts: list[str], languages: list[str] = None) -> list[dict]:
        """
        Classifies and tags a batch of documents (one classifier call per batch).

        Args:
            texts (list[str]): Raw documents.
            languages (list[str], optional): Known languages; detected if omitted.

        Returns:
            list[dict]: Per document: language, category, category_id, confidence,
            tags, entities.
        """
        try:
            if not texts:
                return []
            start = time.time()
            langs = languages or [self._detect(text) for text in texts]
            cleaned = [self._clean(text, lang) for text, lang in zip(texts, langs)]

            probabilities = self.classifier.predict(cleaned)
            predicted = np.argmax(probabilities, axis=1)
            confidences = np.max(probabilities, axis=1)

            results: list[dict] = []
            for i in range(len(texts)):
                label = int(predicted[i])
                # Tagging is best-effort: a missing language model must not break
                # classification.
                try:
                    tag_result = self.tagger.tag(cleaned[i], language=langs[i])
                    tags = tag_result["tags"]
                    entities = tag_result["entities"]
                    keywords = tag_result.get("keywords", [])
                except Exception as te:
                    logger.warning(f"Tagging failed (doc {i}, '{langs[i]}'): {te}")
                    tags, entities, keywords = [], [], []

                # Top-3 predictions for interpretability ("why this category?").
                top_idx = np.argsort(probabilities[i])[::-1][:3]
                top_categories = [
                    {
                        "category": self.label_map.get(int(j), str(int(j))),
                        "probability": round(float(probabilities[i][j]), 4),
                    }
                    for j in top_idx
                ]

                results.append(
                    {
                        "language": langs[i],
                        "category": self.label_map.get(label, str(label)),
                        "category_id": label,
                        "confidence": round(float(confidences[i]), 4),
                        "tags": tags,
                        "keywords": keywords,
                        "entities": entities,
                        "top_categories": top_categories,
                    }
                )

            elapsed = time.time() - start
            logger.info(
                f"Processed {len(texts)} docs in {elapsed:.2f}s "
                f"({len(texts) / max(1e-9, elapsed):.1f} docs/s)."
            )
            return results
        except Exception as e:
            logger.error(f"process_batch failed: {e}")
            raise

    def process(self, text: str, language: str = None) -> dict:
        """Processes a single document (convenience wrapper over ``process_batch``)."""
        try:
            if not text or not str(text).strip():
                raise ValueError("Cannot process an empty document.")
            return self.process_batch([text], [language] if language else None)[0]
        except Exception as e:
            logger.error(f"process failed: {e}")
            raise

    def benchmark(self, texts: list[str], batch_size: int = 32) -> dict:
        """
        Measures end-to-end throughput over ``texts`` (verifies the >=100 docs/sec bar).

        Returns:
            dict: {documents, seconds, docs_per_sec}.
        """
        try:
            start = time.time()
            processed = 0
            for i in range(0, len(texts), batch_size):
                self.process_batch(texts[i : i + batch_size])
                processed += len(texts[i : i + batch_size])
            elapsed = time.time() - start
            docs_per_sec = round(processed / max(1e-9, elapsed), 1)
            logger.info(f"Benchmark: {processed} docs in {elapsed:.2f}s -> {docs_per_sec} docs/s.")
            return {"documents": processed, "seconds": round(elapsed, 2), "docs_per_sec": docs_per_sec}
        except Exception as e:
            logger.error(f"benchmark failed: {e}")
            raise
