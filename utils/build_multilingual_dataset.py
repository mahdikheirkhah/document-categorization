"""
utils/build_multilingual_dataset.py

Offline data-builder for the multi-language corpus (Route A): fetches the English
20 Newsgroups corpus, translates a stratified sample to Swedish and Finnish with
Opus-MT, and persists the combined dataset to ``data/processed_data/``.

Run this as a STANDALONE process, not inside the TensorFlow training kernel. On
macOS, importing TensorFlow and PyTorch into one process triggers a dual-OpenMP
"mutex lock failed" abort, so we force ``USE_TF=0`` here and keep translation in
its own torch-only process. The training notebook then just loads the saved CSV.

Usage:
    python -m utils.build_multilingual_dataset --sample-size 300
"""
import os

# Must be set BEFORE transformers is imported anywhere in this process.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse

import pandas as pd
from loguru import logger

from utils import config
from utils.data_fetcher import (
    EnglishNewsgroupsFetcher,
    SwedishTranslationFetcher,
    FinnishTranslationFetcher,
)
from utils.text_cleaning import LanguageSpecificSanitizer


def _is_translatable(text: str) -> bool:
    """
    True only for genuinely linguistic documents. Filters out the non-text junk in
    20 Newsgroups (ASCII art, symbol/separator lines, single giant tokens) that
    makes the NMT model degenerate into repetition ("^ ^ ^", "# # #", "PRIMA PRIMA").

    Args:
        text (str): A (already cleaned) English document.

    Returns:
        bool: Whether the document is worth translating.
    """
    words = text.split()
    if len(words) < config.MIN_WORDS:
        return False
    alpha_ratio = sum(c.isalpha() for c in text) / max(1, len(text))
    uniq_ratio = len(set(words)) / len(words)
    longest = max(len(w) for w in words)
    return (
        alpha_ratio >= config.MIN_ALPHA_RATIO
        and uniq_ratio >= config.MIN_UNIQUE_WORD_RATIO
        and not (longest > config.MAX_TOKEN_LENGTH and len(words) < 5)
    )

OUTPUT_DIR: str = config.PROCESSED_DATA_DIR
OUTPUT_PATH: str = config.CORPUS_PATH


def build(sample_size: int = None) -> pd.DataFrame:
    """
    Builds and persists the English+Swedish+Finnish corpus.

    Args:
        sample_size (int, optional): Number of English documents to translate per
            language (stratified by label so category balance is preserved). If
            ``None``, the ENTIRE English corpus is translated ("everything").

    Returns:
        pd.DataFrame: The combined multi-language corpus (SCHEMA_COLUMNS).
    """
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        english = EnglishNewsgroupsFetcher().fetch()
        if sample_size is None:
            # "Everything": translate the entire English corpus (slow, offline-scale).
            en_sample = english.reset_index(drop=True)
            logger.info(f"Using the ENTIRE English corpus ({len(en_sample)} docs).")
        else:
            # Stratified n-per-class sampling. (Using frac is fragile: for small
            # sample sizes it rounds down to 0 per class and yields an empty set.)
            n_classes = english["label"].nunique()
            per_class = max(1, sample_size // n_classes)
            en_sample = (
                english.groupby("label", group_keys=False)
                .sample(n=per_class, random_state=config.SEED)
                .reset_index(drop=True)
            )

        # Clean the English source BEFORE translating: stripping headers, quoted
        # replies, and signatures means they are never fed to the translator, so
        # the SV/FI text is cleaner and the model degenerates far less.
        sanitizer = LanguageSpecificSanitizer(config.DEFAULT_LANGUAGE)
        en_sample["text"] = en_sample["text"].fillna("").astype(str).map(sanitizer.clean)

        # Then drop empty/micro AND non-linguistic ("garbage") docs up front, so we
        # never waste translation on them or save degenerate output (ASCII art,
        # symbol lines, single giant tokens that collapse into "^ ^ ^" / "# # #").
        valid_mask = en_sample["text"].map(_is_translatable)
        dropped = int((~valid_mask).sum())
        en_sample = en_sample[valid_mask].reset_index(drop=True)
        logger.info(
            f"Cleaned + filtered: dropped {dropped} low-quality docs; "
            f"translating {len(en_sample)} documents..."
        )

        # Same documents across languages -> one identical label space.
        swedish = SwedishTranslationFetcher(en_sample).fetch()
        finnish = FinnishTranslationFetcher(en_sample).fetch()

        corpus = pd.concat([en_sample, swedish, finnish], ignore_index=True)
        corpus.to_csv(OUTPUT_PATH, index=False)
        logger.info(
            f"Saved {len(corpus)} documents to {OUTPUT_PATH} "
            f"({corpus['language'].value_counts().to_dict()})."
        )
        return corpus
    except Exception as e:
        logger.error(f"Failed to build multilingual dataset: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the multi-language corpus.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Docs per language to translate. Omit to translate the ENTIRE corpus.",
    )
    args = parser.parse_args()
    build(args.sample_size)
