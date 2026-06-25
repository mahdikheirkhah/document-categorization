"""
utils/data_fetcher.py

Data acquisition layer for the multi-language document categorization corpus
(Route A: English 20 Newsgroups + machine-translated Swedish and Finnish).

This module defines one abstract contract, ``BaseDatasetFetcher``, and a family
of concrete fetchers that each return documents in a single standardized schema.
Because every fetcher speaks the same schema, heterogeneous sources (an English
Hugging Face corpus, an OPUS-MT Swedish translation, an OPUS-MT Finnish
translation) can be concatenated into one unified, label-aligned dataset.

OOP design:
    * Abstraction  -> BaseDatasetFetcher defines the ``fetch`` interface.
    * Inheritance  -> BaseDatasetFetcher -> BaseTranslationFetcher -> {Swedish, Finnish}.
    * Encapsulation -> translation pipeline / sampling internals are private helpers.
    * Polymorphism -> MultilingualCorpusFetcher drives any list of BaseDatasetFetcher.
"""

import os
from abc import ABC, abstractmethod

import pandas as pd
from loguru import logger

from utils import config
from utils.data_loader import HuggingFaceCorpusLoader

# Standardized schema every fetcher returns (re-exported from the central config).
SCHEMA_COLUMNS: list[str] = config.SCHEMA_COLUMNS

# OPUS-MT translation models (Helsinki-NLP), sourced from the central config.
OPUS_MODEL_EN_SV: str = config.OPUS_MODELS["sv"]
OPUS_MODEL_EN_FI: str = config.OPUS_MODELS["fi"]


class BaseDatasetFetcher(ABC):
    """
    Abstract base class defining the interface for every component that fetches
    documents and contributes them to the unified multi-language dataset.

    Subclasses encapsulate *where* the documents come from (a remote hub, a
    translation model, a local file) while guaranteeing, via ``fetch``, that the
    output always conforms to ``SCHEMA_COLUMNS``. This enables polymorphic
    assembly of the final corpus.
    """

    def __init__(self, language: str) -> None:
        """
        Initializes the fetcher with the language it is responsible for.

        Args:
            language (str): ISO 639-1 code describing the language this fetcher
                produces (e.g. 'en', 'sv', 'fi').
        """
        try:
            self.language = language
            logger.info(f"Initialized {type(self).__name__} for language '{language}'.")
        except Exception as e:
            logger.error(f"Failed to initialize {type(self).__name__}: {e}")
            raise

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """
        Fetches documents and returns them in the standardized schema.

        Returns:
            pd.DataFrame: A frame containing exactly the ``SCHEMA_COLUMNS``.
        """
        pass

    def _standardize(
        self,
        dataframe: pd.DataFrame,
        text_col: str,
        label_col: str,
        label_text_col: str,
    ) -> pd.DataFrame:
        """
        Shared helper (Template Method) that coerces an arbitrary source frame
        into the canonical ``SCHEMA_COLUMNS`` layout and stamps the language.

        Args:
            dataframe (pd.DataFrame): The raw source frame.
            text_col (str): Name of the document text column in the source.
            label_col (str): Name of the integer label column in the source.
            label_text_col (str): Name of the human-readable category column.

        Returns:
            pd.DataFrame: A frame containing exactly the ``SCHEMA_COLUMNS``.
        """
        try:
            missing = {text_col, label_col, label_text_col} - set(dataframe.columns)
            if missing:
                raise KeyError(f"Source frame is missing required columns: {missing}")

            standardized = pd.DataFrame(
                {
                    "text": dataframe[text_col].astype(str),
                    "label": dataframe[label_col],
                    "label_text": dataframe[label_text_col].astype(str),
                    "language": self.language,
                }
            )
            logger.info(
                f"Standardized {len(standardized)} '{self.language}' documents "
                "into the corpus schema."
            )
            return standardized
        except KeyError as ke:
            logger.error(f"Schema standardization failed: {ke}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during schema standardization: {e}")
            raise


class EnglishNewsgroupsFetcher(BaseDatasetFetcher):
    """
    Fetches the English 20 Newsgroups corpus from Hugging Face.

    This is the canonical labelled source of Route A; its labels are reused
    verbatim by the translation fetchers so that every language shares one
    identical label space.
    """

    def __init__(
        self,
        dataset_name: str = config.DATASET_NAME,
        split: str = config.DATASET_SPLIT,
    ) -> None:
        """
        Args:
            dataset_name (str): Hugging Face dataset identifier.
            split (str): Dataset split to load (e.g. 'train', 'test').
        """
        try:
            super().__init__(language="en")
            self.loader = HuggingFaceCorpusLoader(dataset_name=dataset_name, split=split)
        except Exception as e:
            logger.error(f"Failed to initialize EnglishNewsgroupsFetcher: {e}")
            raise

    def fetch(self) -> pd.DataFrame:
        """
        Downloads 20 Newsgroups and returns it in the standardized schema.

        Returns:
            pd.DataFrame: English documents with the ``SCHEMA_COLUMNS``.
        """
        try:
            raw_dataframe = self.loader.load_data()
            return self._standardize(
                raw_dataframe,
                text_col="text",
                label_col="label",
                label_text_col="label_text",
            )
        except Exception as e:
            logger.error(f"EnglishNewsgroupsFetcher.fetch failed: {e}")
            raise


class BaseTranslationFetcher(BaseDatasetFetcher):
    """
    Intermediate abstract class encapsulating the OPUS-MT translation workflow
    shared by every language-specific translation fetcher.

    It takes an already standardized English frame and produces a parallel frame
    in the target language, preserving the original labels. Concrete subclasses
    only declare *which* target language and model to use.
    """

    def __init__(
        self,
        source_dataframe: pd.DataFrame,
        language: str,
        model_name: str,
        sample_size: int = None,
        batch_size: int = config.TRANSLATION_BATCH_SIZE,
        random_state: int = config.SEED,
        framework: str = None,
        translator=None,
    ) -> None:
        """
        Args:
            source_dataframe (pd.DataFrame): Standardized English frame to translate.
            language (str): Target ISO 639-1 code (e.g. 'sv', 'fi').
            model_name (str): Hugging Face OPUS-MT model identifier.
            sample_size (int, optional): If set, translate a label-stratified
                sample of this size to bound cost. Defaults to None (translate all).
            batch_size (int): Translation batch size for throughput.
            random_state (int): Seed for reproducible sampling.
            framework (str, optional): Backend for the pipeline ('pt' or 'tf').
                Defaults to None (auto-detect: PyTorch if available, else TensorFlow).
            translator (callable, optional): Pre-built translation callable for
                dependency injection in tests. Defaults to None (lazy-loaded).
        """
        try:
            super().__init__(language=language)
            if not isinstance(source_dataframe, pd.DataFrame) or source_dataframe.empty:
                raise ValueError("source_dataframe must be a non-empty DataFrame.")

            self.source_dataframe = source_dataframe
            self.model_name = model_name
            self.sample_size = sample_size
            self.batch_size = batch_size
            self.random_state = random_state
            self.framework = framework
            # Optional dependency injection (testability); lazily loaded otherwise.
            self._translator = translator
        except Exception as e:
            logger.error(f"Failed to initialize {type(self).__name__}: {e}")
            raise

    def _resolve_framework(self) -> str:
        """
        Selects the inference backend: an explicit override if provided, else
        PyTorch when installed, otherwise TensorFlow (this project ships TF only).

        Returns:
            str: 'pt' or 'tf'.
        """
        if self.framework is not None:
            return self.framework
        try:
            import torch  # noqa: F401

            return "pt"
        except Exception:
            return "tf"

    def _load_translator(self):
        """
        Lazily instantiates the heavy OPUS-MT translation pipeline.

        Returns:
            callable: A translation pipeline taking a list[str] of documents.
        """
        try:
            if self._translator is None:
                framework = self._resolve_framework()
                if framework == "pt":
                    # macOS fix: stop transformers from importing TensorFlow next to
                    # PyTorch (two OpenMP runtimes in one process -> "mutex lock
                    # failed" abort). Run translation in a process that does NOT also
                    # load the TF transformers classes (i.e. the offline build step,
                    # not the same kernel that trains the TF classifier).
                    os.environ.setdefault("USE_TF", "0")
                from transformers import pipeline

                logger.info(
                    f"Loading OPUS-MT model '{self.model_name}' (framework={framework})..."
                )
                self._translator = pipeline(
                    "translation", model=self.model_name, framework=framework
                )
            return self._translator
        except Exception as e:
            logger.error(f"Failed to load translation model '{self.model_name}': {e}")
            raise

    def _sample_source(self) -> pd.DataFrame:
        """
        Optionally draws a label-stratified sample to bound translation cost,
        preserving the category balance of the English source.

        Returns:
            pd.DataFrame: The (possibly sampled) source frame.
        """
        try:
            if self.sample_size is None or self.sample_size >= len(self.source_dataframe):
                return self.source_dataframe

            fraction = self.sample_size / len(self.source_dataframe)
            # GroupBy.sample keeps the grouping column (unlike groupby.apply in pandas 3).
            sampled = self.source_dataframe.groupby("label", group_keys=False).sample(
                frac=fraction, random_state=self.random_state
            )
            logger.info(
                f"Sampled {len(sampled)} documents (stratified by label) for translation."
            )
            return sampled
        except Exception as e:
            logger.error(f"Stratified sampling failed: {e}")
            raise

    def _translate_texts(self, texts: list[str]) -> list[str]:
        """
        Translates a list of documents into the target language.

        Args:
            texts (list[str]): English documents to translate.

        Returns:
            list[str]: Translated documents in the target language.
        """
        try:
            translator = self._load_translator()
            outputs = translator(
                texts,
                batch_size=self.batch_size,
                truncation=True,
                max_length=config.TRANSLATION_MAX_LENGTH,
            )
            return [item["translation_text"] for item in outputs]
        except Exception as e:
            logger.error(f"Translation to '{self.language}' failed: {e}")
            raise

    def fetch(self) -> pd.DataFrame:
        """
        Translates the (sampled) English source into the target language while
        preserving labels, returning the standardized schema.

        Returns:
            pd.DataFrame: Target-language documents with the ``SCHEMA_COLUMNS``.
        """
        try:
            source = self._sample_source().copy()
            logger.info(
                f"Translating {len(source)} documents EN -> {self.language.upper()}..."
            )
            source["text"] = self._translate_texts(source["text"].astype(str).tolist())
            source["language"] = self.language
            translated = source[SCHEMA_COLUMNS].reset_index(drop=True)
            logger.info(
                f"Produced {len(translated)} '{self.language}' documents via translation."
            )
            return translated
        except Exception as e:
            logger.error(f"{type(self).__name__}.fetch failed: {e}")
            raise


class SwedishTranslationFetcher(BaseTranslationFetcher):
    """Produces Swedish documents by translating the English source (EN -> SV)."""

    def __init__(
        self,
        source_dataframe: pd.DataFrame,
        sample_size: int = None,
        batch_size: int = config.TRANSLATION_BATCH_SIZE,
        random_state: int = config.SEED,
        model_name: str = OPUS_MODEL_EN_SV,
        translator=None,
    ) -> None:
        super().__init__(
            source_dataframe=source_dataframe,
            language="sv",
            model_name=model_name,
            sample_size=sample_size,
            batch_size=batch_size,
            random_state=random_state,
            translator=translator,
        )


class FinnishTranslationFetcher(BaseTranslationFetcher):
    """Produces Finnish documents by translating the English source (EN -> FI)."""

    def __init__(
        self,
        source_dataframe: pd.DataFrame,
        sample_size: int = None,
        batch_size: int = config.TRANSLATION_BATCH_SIZE,
        random_state: int = config.SEED,
        model_name: str = OPUS_MODEL_EN_FI,
        translator=None,
    ) -> None:
        super().__init__(
            source_dataframe=source_dataframe,
            language="fi",
            model_name=model_name,
            sample_size=sample_size,
            batch_size=batch_size,
            random_state=random_state,
            translator=translator,
        )


class MultilingualCorpusFetcher:
    """
    Orchestrator that polymorphically drives a collection of
    ``BaseDatasetFetcher`` instances and concatenates their output into the final
    multi-language corpus that is persisted under ``data/``.

    It depends only on the abstract ``BaseDatasetFetcher`` interface, so new
    languages or sources can be added without changing this class.
    """

    def __init__(self, fetchers: list[BaseDatasetFetcher]) -> None:
        """
        Args:
            fetchers (list[BaseDatasetFetcher]): The per-language fetchers whose
                output will be concatenated into one corpus.
        """
        try:
            if not fetchers:
                raise ValueError("MultilingualCorpusFetcher requires at least one fetcher.")
            self.fetchers = fetchers
            logger.info(
                f"Initialized MultilingualCorpusFetcher with {len(fetchers)} source(s)."
            )
        except Exception as e:
            logger.error(f"Failed to initialize MultilingualCorpusFetcher: {e}")
            raise

    def build(self) -> pd.DataFrame:
        """
        Runs every fetcher and concatenates the results into the unified corpus.

        Returns:
            pd.DataFrame: The combined multi-language corpus (``SCHEMA_COLUMNS``).
        """
        try:
            frames: list[pd.DataFrame] = []
            for fetcher in self.fetchers:
                frames.append(fetcher.fetch())  # polymorphic call on the base type

            corpus = pd.concat(frames, ignore_index=True)
            languages = sorted(corpus["language"].unique().tolist())
            logger.info(
                f"Assembled multi-language corpus: {len(corpus)} documents "
                f"across languages {languages}."
            )
            return corpus
        except Exception as e:
            logger.error(f"Failed to assemble multi-language corpus: {e}")
            raise
