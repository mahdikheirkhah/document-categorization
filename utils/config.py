"""
utils/config.py — single source of truth for the pipeline (Ablation-Driven Architecture).

Every tunable value used across the data, model, tagging, and serving pipeline lives
here. An ablation is therefore run by changing a value in this one file (or by
overriding the matching argument), never by editing logic across modules. Keep ALL
pipeline constants here.

Paths are anchored to the repository root (this file's parent's parent), so they
resolve correctly no matter the current working directory.
"""

import os

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED: int = 42  # numpy / tensorflow / langdetect / sampling / train-test split

# --------------------------------------------------------------------------- #
# Languages
# --------------------------------------------------------------------------- #
SUPPORTED_LANGUAGES: list[str] = ["en", "sv", "fi"]
DEFAULT_LANGUAGE: str = "en"

# --------------------------------------------------------------------------- #
# Paths (anchored to the repository root)
# --------------------------------------------------------------------------- #
ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR: str = os.path.join(ROOT_DIR, "data")
PROCESSED_DATA_DIR: str = os.path.join(DATA_DIR, "processed_data")
CORPUS_PATH: str = os.path.join(PROCESSED_DATA_DIR, "multilingual_corpus.csv")
CHECKPOINT_DIR: str = os.path.join(ROOT_DIR, "models", "checkpoints")
REPORTS_DIR: str = os.path.join(ROOT_DIR, "reports")

BEST_WEIGHTS_FILENAME: str = "text_classifier_best.h5"
MODEL_CONFIG_FILENAME: str = "config.json"
HISTORY_FILENAME: str = "training_history.csv"
METRICS_PATH: str = os.path.join(REPORTS_DIR, "performance_metrics.json")
EXAMPLE_PREDICTIONS_PATH: str = os.path.join(REPORTS_DIR, "example_predictions.csv")

# --------------------------------------------------------------------------- #
# Dataset & standardized schema
# --------------------------------------------------------------------------- #
# MASSIVE (Amazon) is a natively multilingual, parallel dataset — the same
# utterances are provided in every language, so all languages share one identical
# label space with no translation step. The canonical "AmazonScience/massive"
# repo ships a dataset *script*, which datasets>=4 no longer executes, so we load
# the parquet mirror published by MTEB. We classify the 18 coarse "scenario"
# domains (alarm, calendar, weather, ...) rather than the 60 fine intents: fewer,
# better-separated classes make the >=85% accuracy / >=0.80 macro-F1 targets
# realistic, while still clearing the ">=5 categories" requirement.
MASSIVE_DATASET: str = "mteb/amazon_massive_scenario"
# One config per language; in MASSIVE the config name is just the ISO code.
MASSIVE_LANGUAGE_CONFIGS: dict[str, str] = {"en": "en", "sv": "sv", "fi": "fi"}
# Splits pulled per language and concatenated into the corpus. MASSIVE is
# parallel, so each split holds the same utterances across all languages.
MASSIVE_SPLITS: list[str] = ["train"]
# Column names in the MTEB parquet: the utterance text and the scenario name.
MASSIVE_TEXT_COL: str = "text"
MASSIVE_LABEL_COL: str = "label_text"  # human-readable scenario name (e.g. "alarm")

SCHEMA_COLUMNS: list[str] = ["text", "label", "label_text", "language"]

# --------------------------------------------------------------------------- #
# Corpus quality filter & train/val split
# --------------------------------------------------------------------------- #
# MASSIVE utterances are short, already-clean commands (e.g. "play music"), so
# the only filter is dropping empty / whitespace-only ("ghost") rows.
MIN_WORDS: int = 1  # drop only empty docs (MASSIVE utterances are short)
TEST_SIZE: float = 0.2  # validation fraction of the stratified split

# --------------------------------------------------------------------------- #
# Classifier (TF-IDF baseline + DistilBERT)
# --------------------------------------------------------------------------- #
BASELINE_MAX_FEATURES: int = 10000
DISTILBERT_MODEL: str = "distilbert-base-multilingual-cased"
MAX_LENGTH: int = 256
LEARNING_RATE: float = 3e-5
EPOCHS: int = 5
TRAIN_BATCH_SIZE: int = 16
INFERENCE_BATCH_SIZE: int = 32

# --------------------------------------------------------------------------- #
# Tokenizer (Finnish sub-word fallback)
# --------------------------------------------------------------------------- #
FINNISH_SUBWORD_MODEL: str = "distilbert-base-multilingual-cased"

# --------------------------------------------------------------------------- #
# Tagger (SpaCy NER + dynamic keywords)
# --------------------------------------------------------------------------- #
SPACY_MODELS: dict[str, str] = {
    "en": "en_core_web_sm",
    "sv": "sv_core_news_sm",
    "fi": "fi_core_news_sm",
}
# Normalize heterogeneous SpaCy entity labels (en/fi OntoNotes; sv SUC) -> one scheme.
ENTITY_LABEL_TO_TAG: dict[str, str] = {
    "PERSON": "person",
    "PER": "person",
    "PRS": "person",
    "ORG": "organization",
    "GPE": "location",
    "LOC": "location",
    "FAC": "location",
    "NORP": "group",
    "DATE": "date",
    "TIME": "date",
    "TME": "date",
    "MONEY": "money",
    "PERCENT": "metric",
    "QUANTITY": "metric",
    "MSR": "metric",
    "CARDINAL": "number",
    "ORDINAL": "number",
    "PRODUCT": "product",
    "OBJ": "product",
    "EVENT": "event",
    "EVN": "event",
    "WORK_OF_ART": "work",
    "WRK": "work",
    "LAW": "law",
    "LANGUAGE": "language",
    "MISC": "misc",
}
KEYWORD_POS: set[str] = {"NOUN", "PROPN"}  # parts of speech kept as content keywords
MAX_KEYWORDS: int = 8

# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
# Below this top-class probability the document is likely out-of-domain, so the
# predicted category should be treated as unreliable.
LOW_CONFIDENCE_THRESHOLD: float = 0.5
