"""
utils/build_multilingual_dataset.py

Offline data-builder for the multi-language corpus. Loads the **MASSIVE**
scenario dataset (Amazon) for English, Swedish and Finnish, applies light
language-specific cleaning, and persists the combined dataset to
``data/processed_data/``.

MASSIVE is natively multilingual and parallel, so Swedish and Finnish are real
native text (not machine translations) and every language shares one identical
integer label space. The English fetcher establishes the canonical
scenario->id map, which is injected into the Swedish and Finnish fetchers so the
three frames align perfectly.

The training notebook / ``train.py`` then just load the saved CSV.

Usage:
    python -m utils.build_multilingual_dataset                 # full corpus
    python -m utils.build_multilingual_dataset --sample-size 1200   # quick build
"""

import os

# Keep the HF tokenizers backend quiet in this short, single-threaded build.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse

import pandas as pd
from loguru import logger

from utils import config
from utils.data_fetcher import MassiveScenarioFetcher, MultilingualCorpusFetcher
from utils.text_cleaning import LanguageSpecificSanitizer

OUTPUT_DIR: str = config.PROCESSED_DATA_DIR
OUTPUT_PATH: str = config.CORPUS_PATH
SAMPLE_PATH: str = os.path.join(OUTPUT_DIR, "multilingual_sample.csv")
SAMPLE_ROWS_PER_LANGUAGE: int = 100  # tiny demo/EDA slice written alongside the corpus


def _clean_per_language(corpus: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the polymorphic, language-specific cleaning pipeline to each language
    partition (Unicode normalization that preserves å/ä/ö, diacritic protection
    for Swedish, suffix-preserving handling for Finnish, whitespace normalization).

    Args:
        corpus (pd.DataFrame): Combined corpus with the standardized schema.

    Returns:
        pd.DataFrame: The corpus with a cleaned ``text`` column.
    """
    try:
        cleaned_parts: list[pd.DataFrame] = []
        for language, group in corpus.groupby("language", sort=False):
            sanitizer = LanguageSpecificSanitizer(str(language))
            part = group.copy()
            part["text"] = part["text"].fillna("").astype(str).map(sanitizer.clean)
            cleaned_parts.append(part)
            logger.info(f"Cleaned {len(part)} '{language}' documents.")
        return pd.concat(cleaned_parts, ignore_index=True)
    except Exception as e:
        logger.error(f"Per-language cleaning failed: {e}")
        raise


def _drop_ghost_documents(corpus: pd.DataFrame) -> pd.DataFrame:
    """
    Drops empty / whitespace-only ("ghost") rows produced by cleaning, using the
    ``MIN_WORDS`` threshold from the central config.

    Args:
        corpus (pd.DataFrame): The cleaned corpus.

    Returns:
        pd.DataFrame: The corpus with ghost documents removed.
    """
    try:
        word_counts = corpus["text"].fillna("").str.split().map(len)
        keep_mask = word_counts >= config.MIN_WORDS
        dropped = int((~keep_mask).sum())
        if dropped:
            logger.warning(f"Dropped {dropped} empty/whitespace-only documents.")
        return corpus[keep_mask].reset_index(drop=True)
    except Exception as e:
        logger.error(f"Ghost-document filtering failed: {e}")
        raise


def _write_sample(corpus: pd.DataFrame) -> None:
    """
    Writes a tiny, label-stratified per-language slice for fast EDA and demos.

    Args:
        corpus (pd.DataFrame): The full cleaned corpus.
    """
    try:
        parts = [
            group.sample(
                n=min(SAMPLE_ROWS_PER_LANGUAGE, len(group)),
                random_state=config.SEED,
            )
            for _, group in corpus.groupby("language", sort=False)
        ]
        sample = pd.concat(parts, ignore_index=True)
        sample.to_csv(SAMPLE_PATH, index=False)
        logger.info(f"Saved {len(sample)}-row demo sample to {SAMPLE_PATH}.")
    except Exception as e:
        logger.error(f"Failed to write demo sample: {e}")
        raise


def build(sample_size: int = None) -> pd.DataFrame:
    """
    Builds and persists the English + Swedish + Finnish MASSIVE corpus.

    Args:
        sample_size (int, optional): Rows per language to keep (label-stratified)
            for a fast, small build. If ``None``, the entire configured split is
            used for every language.

    Returns:
        pd.DataFrame: The combined, cleaned multi-language corpus (SCHEMA_COLUMNS).
    """
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # English establishes the canonical scenario -> integer-id label map, which
        # is shared with Swedish and Finnish so all languages align on one space.
        # (The repeated English load is served from the local HF cache, so it is
        # cheap and keeps every language flowing through the same orchestrator.)
        shared_label_map = MassiveScenarioFetcher(
            "en", sample_size=sample_size
        ).fetch_label_map()
        logger.info(
            f"Established shared label space with {len(shared_label_map)} scenarios."
        )

        fetchers = [
            MassiveScenarioFetcher(
                language, label_to_id=shared_label_map, sample_size=sample_size
            )
            for language in config.SUPPORTED_LANGUAGES
        ]
        corpus = MultilingualCorpusFetcher(fetchers).build()

        corpus = _clean_per_language(corpus)
        corpus = _drop_ghost_documents(corpus)

        corpus.to_csv(OUTPUT_PATH, index=False)
        logger.info(
            f"Saved {len(corpus)} documents to {OUTPUT_PATH} "
            f"({corpus['language'].value_counts().to_dict()})."
        )

        _write_sample(corpus)
        return corpus
    except Exception as e:
        logger.error(f"Failed to build multilingual dataset: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the MASSIVE multi-language corpus."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Rows per language to keep (stratified). Omit to use the full split.",
    )
    args = parser.parse_args()
    build(args.sample_size)
