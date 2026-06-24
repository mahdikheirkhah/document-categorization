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

from utils.data_fetcher import (
    EnglishNewsgroupsFetcher,
    SwedishTranslationFetcher,
    FinnishTranslationFetcher,
)

OUTPUT_DIR: str = "data/processed_data"
OUTPUT_PATH: str = os.path.join(OUTPUT_DIR, "multilingual_corpus.csv")


def build(sample_size: int) -> pd.DataFrame:
    """
    Builds and persists the English+Swedish+Finnish corpus.

    Args:
        sample_size (int): Number of English documents to translate per language
            (stratified by label so category balance is preserved).

    Returns:
        pd.DataFrame: The combined multi-language corpus (SCHEMA_COLUMNS).
    """
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        english = EnglishNewsgroupsFetcher().fetch()
        fraction = min(1.0, sample_size / len(english))
        en_sample = (
            english.groupby("label", group_keys=False)
            .sample(frac=fraction, random_state=42)
            .reset_index(drop=True)
        )

        # Drop NLP-missing docs BEFORE translating. An empty/whitespace ("ghost")
        # or micro document makes the NMT model hallucinate (e.g. an empty input
        # becomes Finnish "- Ei, ei, ei, ei..."), so we remove them up front. This
        # also guarantees the saved corpus has no empty/NaN text on reload.
        stripped = en_sample["text"].fillna("").astype(str).str.strip()
        valid_mask = stripped.str.split().map(len) >= 3
        dropped = int((~valid_mask).sum())
        en_sample = en_sample[valid_mask].reset_index(drop=True)
        logger.info(
            f"Dropped {dropped} empty/micro docs; translating {len(en_sample)} documents..."
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
        default=300,
        help="Number of English documents to translate per language.",
    )
    args = parser.parse_args()
    build(args.sample_size)
