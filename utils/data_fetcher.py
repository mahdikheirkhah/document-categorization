"""
utils/data_fetcher.py

Data acquisition layer for the multi-language document categorization corpus.

We use the **MASSIVE** dataset (Amazon), a natively multilingual, *parallel*
corpus: the same utterances are provided in every language, each labelled with
one of 18 coarse "scenario" domains (alarm, calendar, weather, ...). Because the
languages are parallel, every language shares one identical label space with no
translation step — Swedish and Finnish are real, native text rather than machine
translations of English.

This module defines one abstract contract, ``BaseDatasetFetcher``, and a concrete
``MassiveScenarioFetcher`` that returns documents in a single standardized schema.
Because every fetcher speaks the same schema, the per-language frames concatenate
into one unified, label-aligned dataset via ``MultilingualCorpusFetcher``.

OOP design:
    * Abstraction   -> BaseDatasetFetcher defines the ``fetch`` interface.
    * Inheritance   -> MassiveScenarioFetcher extends BaseDatasetFetcher.
    * Encapsulation -> raw loading, sampling and label-mapping are private helpers.
    * Polymorphism  -> MultilingualCorpusFetcher drives any list of BaseDatasetFetcher.
"""

from abc import ABC, abstractmethod
from typing import Callable

import pandas as pd
from loguru import logger

from utils import config
from utils.data_loader import HuggingFaceCorpusLoader

# Standardized schema every fetcher returns (re-exported from the central config).
SCHEMA_COLUMNS: list[str] = config.SCHEMA_COLUMNS


def build_scenario_label_map(label_names: list[str]) -> dict[str, int]:
    """
    Builds a deterministic, language-agnostic mapping from a scenario name to an
    integer id by sorting the unique names alphabetically.

    Sorting makes the mapping reproducible and identical across languages (MASSIVE
    exposes the same scenario set in every language), so the Swedish, Finnish and
    English frames end up with one shared integer label space.

    Args:
        label_names (list[str]): Scenario names observed in the data (may repeat).

    Returns:
        dict[str, int]: Mapping ``{scenario_name: integer_id}``.
    """
    try:
        unique_sorted = sorted({str(name) for name in label_names})
        if not unique_sorted:
            raise ValueError("Cannot build a label map from zero scenario names.")
        return {name: index for index, name in enumerate(unique_sorted)}
    except Exception as e:
        logger.error(f"Failed to build scenario label map: {e}")
        raise


class BaseDatasetFetcher(ABC):
    """
    Abstract base class defining the interface for every component that fetches
    documents and contributes them to the unified multi-language dataset.

    Subclasses encapsulate *where* the documents come from while guaranteeing, via
    ``fetch``, that the output always conforms to ``SCHEMA_COLUMNS``. This enables
    polymorphic assembly of the final corpus.
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


class MassiveScenarioFetcher(BaseDatasetFetcher):
    """
    Fetches one language of the MASSIVE *scenario* dataset from Hugging Face and
    returns it in the standardized corpus schema.

    The raw MASSIVE scenario label is a string (e.g. ``"alarm"``); it is mapped to
    a stable integer id via :func:`build_scenario_label_map`. To guarantee that
    every language shares the same integer label space, callers (e.g. the corpus
    builder) compute the map once and inject it through ``label_to_id``.
    """

    def __init__(
        self,
        language: str,
        dataset_name: str = config.MASSIVE_DATASET,
        splits: list[str] = None,
        label_to_id: dict[str, int] = None,
        sample_size: int = None,
        random_state: int = config.SEED,
        loader: Callable[[str, str, str], pd.DataFrame] = None,
    ) -> None:
        """
        Args:
            language (str): Target ISO 639-1 code ('en', 'sv', 'fi').
            dataset_name (str): Hugging Face dataset identifier (MTEB MASSIVE mirror).
            splits (list[str], optional): MASSIVE splits to load and concatenate.
                Defaults to ``config.MASSIVE_SPLITS``.
            label_to_id (dict[str, int], optional): Pre-computed scenario->id map to
                share one label space across languages. If ``None`` it is derived
                from this language's own data during ``fetch``.
            sample_size (int, optional): If set, keep a label-stratified sample of
                this many rows (useful for fast, small builds). Defaults to all.
            random_state (int): Seed for reproducible sampling.
            loader (callable, optional): Dependency-injection seam for tests; a
                callable ``(dataset_name, language_config, split) -> pd.DataFrame``.
                Defaults to loading via :class:`HuggingFaceCorpusLoader`.
        """
        try:
            super().__init__(language=language)
            if language not in config.MASSIVE_LANGUAGE_CONFIGS:
                raise ValueError(
                    f"Unsupported language '{language}'. "
                    f"Expected one of {list(config.MASSIVE_LANGUAGE_CONFIGS)}."
                )
            self.dataset_name = dataset_name
            self.language_config = config.MASSIVE_LANGUAGE_CONFIGS[language]
            self.splits = splits if splits is not None else config.MASSIVE_SPLITS
            self.label_to_id = label_to_id
            self.sample_size = sample_size
            self.random_state = random_state
            self._loader = loader
        except Exception as e:
            logger.error(f"Failed to initialize {type(self).__name__}: {e}")
            raise

    def _load_split(self, split: str) -> pd.DataFrame:
        """
        Loads a single MASSIVE split for this language.

        Args:
            split (str): Split name ('train', 'validation', 'test').

        Returns:
            pd.DataFrame: The raw split as a DataFrame.
        """
        try:
            if self._loader is not None:
                return self._loader(self.dataset_name, self.language_config, split)
            corpus_loader = HuggingFaceCorpusLoader(
                dataset_name=self.dataset_name,
                split=split,
                subset=self.language_config,
            )
            return corpus_loader.load_data()
        except Exception as e:
            logger.error(
                f"Failed to load MASSIVE split '{split}' for '{self.language}': {e}"
            )
            raise

    def _load_raw(self) -> pd.DataFrame:
        """
        Loads and concatenates every configured split for this language.

        Returns:
            pd.DataFrame: The combined raw frame for this language.
        """
        try:
            frames = [self._load_split(split) for split in self.splits]
            raw = pd.concat(frames, ignore_index=True)
            logger.info(
                f"Loaded {len(raw)} raw '{self.language}' MASSIVE rows "
                f"from splits {self.splits}."
            )
            return raw
        except Exception as e:
            logger.error(f"Failed to load raw MASSIVE data for '{self.language}': {e}")
            raise

    def _maybe_sample(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Optionally draws a label-stratified sample to bound corpus size, preserving
        the scenario balance of the source.

        Args:
            raw (pd.DataFrame): The full raw frame.

        Returns:
            pd.DataFrame: The (possibly sampled) frame.
        """
        try:
            if self.sample_size is None or self.sample_size >= len(raw):
                return raw
            n_classes = max(1, raw[config.MASSIVE_LABEL_COL].nunique())
            per_class = max(1, self.sample_size // n_classes)
            sampled = (
                raw.groupby(config.MASSIVE_LABEL_COL, group_keys=False)
                .sample(n=per_class, random_state=self.random_state)
                .reset_index(drop=True)
            )
            logger.info(
                f"Sampled {len(sampled)} '{self.language}' rows "
                f"(stratified by scenario, ~{per_class}/class)."
            )
            return sampled
        except Exception as e:
            logger.error(f"Stratified sampling failed for '{self.language}': {e}")
            raise

    def fetch_label_map(self) -> dict[str, int]:
        """
        Loads this language and returns the canonical scenario->id map without a
        full standardize pass. Used to establish one shared label space before
        fetching every language.

        Returns:
            dict[str, int]: Mapping ``{scenario_name: integer_id}``.
        """
        try:
            raw = self._load_raw()
            if config.MASSIVE_LABEL_COL not in raw.columns:
                raise KeyError(
                    f"MASSIVE frame missing label column "
                    f"'{config.MASSIVE_LABEL_COL}'; found {list(raw.columns)}."
                )
            self.label_to_id = build_scenario_label_map(
                raw[config.MASSIVE_LABEL_COL].tolist()
            )
            return self.label_to_id
        except Exception as e:
            logger.error(f"fetch_label_map failed for '{self.language}': {e}")
            raise

    def fetch(self) -> pd.DataFrame:
        """
        Loads MASSIVE for this language, maps scenario names to a shared integer
        label space, and returns the standardized schema.

        Returns:
            pd.DataFrame: Documents with exactly the ``SCHEMA_COLUMNS``.
        """
        try:
            raw = self._maybe_sample(self._load_raw())

            label_col = config.MASSIVE_LABEL_COL
            text_col = config.MASSIVE_TEXT_COL
            if label_col not in raw.columns or text_col not in raw.columns:
                raise KeyError(
                    f"MASSIVE frame missing required columns "
                    f"'{text_col}'/'{label_col}'; found {list(raw.columns)}."
                )

            # Establish (or reuse) the shared scenario -> integer-id mapping.
            if self.label_to_id is None:
                self.label_to_id = build_scenario_label_map(raw[label_col].tolist())

            scenario_names = raw[label_col].astype(str)
            unknown = set(scenario_names) - set(self.label_to_id)
            if unknown:
                raise KeyError(f"Scenarios absent from the shared label map: {unknown}")

            prepared = pd.DataFrame(
                {
                    text_col: raw[text_col].astype(str),
                    "label_id": scenario_names.map(self.label_to_id).astype(int),
                    "label_name": scenario_names,
                }
            )
            return self._standardize(
                prepared,
                text_col=text_col,
                label_col="label_id",
                label_text_col="label_name",
            )
        except Exception as e:
            logger.error(
                f"{type(self).__name__}.fetch failed for '{self.language}': {e}"
            )
            raise


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
                raise ValueError(
                    "MultilingualCorpusFetcher requires at least one fetcher."
                )
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
