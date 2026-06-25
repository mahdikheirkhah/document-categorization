import pytest
import pandas as pd

from utils.data_fetcher import (
    BaseDatasetFetcher,
    SwedishTranslationFetcher,
    FinnishTranslationFetcher,
    MultilingualCorpusFetcher,
    SCHEMA_COLUMNS,
)


def _english_frame() -> pd.DataFrame:
    """A tiny, already-standardized English frame to translate from."""
    return pd.DataFrame(
        {
            "text": ["Stocks fell today.", "The team won the match.", "A new vaccine was approved."],
            "label": [0, 1, 2],
            "label_text": ["finance", "sport", "health"],
            "language": ["en", "en", "en"],
        }
    )


def _fake_translator(prefix: str):
    """Returns a callable mimicking a transformers translation pipeline (no model download)."""

    def _translate(texts, **kwargs):
        return [{"translation_text": f"{prefix}:{text}"} for text in texts]

    return _translate


class _DummyFetcher(BaseDatasetFetcher):
    """Minimal concrete subclass used to exercise the shared _standardize helper."""

    def __init__(self) -> None:
        super().__init__(language="en")

    def fetch(self) -> pd.DataFrame:
        raw = pd.DataFrame({"body": ["hello world"], "y": [0], "cat": ["greeting"]})
        return self._standardize(raw, text_col="body", label_col="y", label_text_col="cat")


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


def test_swedish_translation_preserves_labels() -> None:
    """EN->SV translation keeps the label space identical and stamps language='sv'."""
    english = _english_frame()
    fetcher = SwedishTranslationFetcher(
        source_dataframe=english, translator=_fake_translator("SV")
    )
    swedish = fetcher.fetch()

    assert list(swedish.columns) == SCHEMA_COLUMNS
    assert (swedish["language"] == "sv").all()
    assert swedish["label"].tolist() == english["label"].tolist()
    assert swedish["label_text"].tolist() == english["label_text"].tolist()
    assert swedish.loc[0, "text"].startswith("SV:")


def test_finnish_translation_language_code() -> None:
    """EN->FI translation stamps language='fi' and routes through the FI fetcher."""
    english = _english_frame()
    fetcher = FinnishTranslationFetcher(
        source_dataframe=english, translator=_fake_translator("FI")
    )
    finnish = fetcher.fetch()

    assert (finnish["language"] == "fi").all()
    assert finnish.loc[0, "text"].startswith("FI:")


def test_translation_requires_non_empty_source() -> None:
    """An empty source frame triggers a controlled ValueError at construction."""
    with pytest.raises(ValueError):
        SwedishTranslationFetcher(source_dataframe=pd.DataFrame())


def test_multilingual_corpus_concatenation() -> None:
    """The orchestrator polymorphically combines EN + SV + FI into one corpus."""
    english = _english_frame()

    class _EnFetcher(BaseDatasetFetcher):
        def __init__(self) -> None:
            super().__init__(language="en")

        def fetch(self) -> pd.DataFrame:
            return english

    swedish = SwedishTranslationFetcher(
        source_dataframe=english, translator=_fake_translator("SV")
    )
    finnish = FinnishTranslationFetcher(
        source_dataframe=english, translator=_fake_translator("FI")
    )

    corpus = MultilingualCorpusFetcher([_EnFetcher(), swedish, finnish]).build()

    assert len(corpus) == len(english) * 3
    assert sorted(corpus["language"].unique().tolist()) == ["en", "fi", "sv"]
    assert list(corpus.columns) == SCHEMA_COLUMNS


def test_stratified_sampling_preserves_columns() -> None:
    """sample_size triggers stratified sampling that must keep every column
    (regression: groupby.apply silently dropped the grouping column in pandas 3)."""
    english = pd.concat([_english_frame()] * 10, ignore_index=True)  # 30 rows, 3 labels
    fetcher = SwedishTranslationFetcher(
        source_dataframe=english, sample_size=9, translator=_fake_translator("SV")
    )
    sampled = fetcher._sample_source()
    assert set(sampled.columns) == set(english.columns)
    assert "label" in sampled.columns
    assert 0 < len(sampled) <= len(english)
