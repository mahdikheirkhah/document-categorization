import pytest
import pandas as pd

from utils import config
from utils.data_fetcher import (
    BaseDatasetFetcher,
    MassiveScenarioFetcher,
    MultilingualCorpusFetcher,
    build_scenario_label_map,
    SCHEMA_COLUMNS,
)


def _fake_massive_frame(language_config: str) -> pd.DataFrame:
    """
    A tiny MASSIVE-like frame mimicking the MTEB parquet schema for one language.

    Two scenarios ('alarm', 'weather') x three rows. The scenario names are
    identical across languages (MASSIVE is parallel); only the text differs.
    """
    scenarios = ["weather", "alarm", "weather", "alarm", "weather", "alarm"]
    return pd.DataFrame(
        {
            "id": [str(i) for i in range(len(scenarios))],
            "label": scenarios,
            "label_text": scenarios,
            "text": [f"{language_config} utterance {i}" for i in range(len(scenarios))],
            "lang": language_config,
        }
    )


def _fake_loader(dataset_name: str, language_config: str, split: str) -> pd.DataFrame:
    """Stand-in for HuggingFaceCorpusLoader — returns a frame with no network call."""
    return _fake_massive_frame(language_config)


class _DummyFetcher(BaseDatasetFetcher):
    """Minimal concrete subclass used to exercise the shared _standardize helper."""

    def __init__(self) -> None:
        super().__init__(language="en")

    def fetch(self) -> pd.DataFrame:
        raw = pd.DataFrame({"body": ["hello world"], "y": [0], "cat": ["greeting"]})
        return self._standardize(
            raw, text_col="body", label_col="y", label_text_col="cat"
        )


def test_base_fetcher_is_abstract() -> None:
    """The abstract base cannot be instantiated directly (enforces the interface)."""
    with pytest.raises(TypeError):
        BaseDatasetFetcher(language="en")


def test_standardize_produces_schema() -> None:
    """A concrete fetcher coerces arbitrary columns into the canonical schema."""
    df = _DummyFetcher().fetch()
    assert list(df.columns) == SCHEMA_COLUMNS
    assert df.loc[0, "language"] == "en"
    assert df.loc[0, "text"] == "hello world"


def test_standardize_missing_column_raises() -> None:
    """Standardization fails loudly (KeyError) when a required column is absent."""
    fetcher = _DummyFetcher()
    bad = pd.DataFrame({"body": ["only text, no label columns"]})
    with pytest.raises(KeyError):
        fetcher._standardize(bad, "body", "y", "cat")


def test_build_scenario_label_map_is_sorted_and_deterministic() -> None:
    """Scenario names map to ids in alphabetical order, identically every time."""
    mapping = build_scenario_label_map(["weather", "alarm", "weather", "calendar"])
    assert mapping == {"alarm": 0, "calendar": 1, "weather": 2}
    # Same names in a different order yield the same mapping (language parity).
    assert build_scenario_label_map(["calendar", "alarm", "weather"]) == mapping


def test_build_scenario_label_map_empty_raises() -> None:
    """An empty name list triggers a controlled ValueError."""
    with pytest.raises(ValueError):
        build_scenario_label_map([])


def test_massive_fetch_produces_schema_with_integer_labels() -> None:
    """The MASSIVE fetcher returns the canonical schema with integer labels."""
    fetcher = MassiveScenarioFetcher("en", loader=_fake_loader)
    frame = fetcher.fetch()
    assert list(frame.columns) == SCHEMA_COLUMNS
    assert (frame["language"] == "en").all()
    assert pd.api.types.is_integer_dtype(frame["label"])
    # 'alarm' sorts before 'weather' -> alarm=0, weather=1.
    alarm_labels = frame.loc[frame["label_text"] == "alarm", "label"].unique().tolist()
    assert alarm_labels == [0]


def test_massive_label_space_is_shared_across_languages() -> None:
    """Injecting the English label map keeps sv/fi ids aligned to the same space."""
    english = MassiveScenarioFetcher("en", loader=_fake_loader)
    shared = english.fetch_label_map()

    swedish = MassiveScenarioFetcher(
        "sv", label_to_id=shared, loader=_fake_loader
    ).fetch()
    finnish = MassiveScenarioFetcher(
        "fi", label_to_id=shared, loader=_fake_loader
    ).fetch()

    for frame in (swedish, finnish):
        pairs = frame[["label", "label_text"]].drop_duplicates()
        mapping = {row.label_text: row.label for row in pairs.itertuples(index=False)}
        assert mapping == shared


def test_massive_unsupported_language_raises() -> None:
    """A language absent from the config raises a controlled ValueError."""
    with pytest.raises(ValueError):
        MassiveScenarioFetcher("es", loader=_fake_loader)


def test_massive_stratified_sampling_preserves_classes() -> None:
    """sample_size draws a stratified slice keeping every scenario and the schema."""
    fetcher = MassiveScenarioFetcher("en", sample_size=2, loader=_fake_loader)
    frame = fetcher.fetch()
    assert list(frame.columns) == SCHEMA_COLUMNS
    assert set(frame["label_text"]) == {"alarm", "weather"}
    assert 0 < len(frame) <= len(_fake_massive_frame("en"))


def test_multilingual_corpus_concatenation() -> None:
    """The orchestrator polymorphically combines EN + SV + FI into one corpus."""
    english = MassiveScenarioFetcher("en", loader=_fake_loader)
    shared = english.fetch_label_map()
    fetchers = [
        MassiveScenarioFetcher(lang, label_to_id=shared, loader=_fake_loader)
        for lang in config.SUPPORTED_LANGUAGES
    ]

    corpus = MultilingualCorpusFetcher(fetchers).build()

    assert sorted(corpus["language"].unique().tolist()) == ["en", "fi", "sv"]
    assert list(corpus.columns) == SCHEMA_COLUMNS
    assert len(corpus) == len(_fake_massive_frame("en")) * 3


def test_multilingual_corpus_requires_a_fetcher() -> None:
    """Building with no fetchers raises a controlled ValueError."""
    with pytest.raises(ValueError):
        MultilingualCorpusFetcher([])
