import pytest
from utils.text_cleaning import (
    HtmlCleaner,
    ArtifactCleaner,
    UnicodeNormalizer,
    DuplicationCleaner,
    TextCleaningPipeline,
)


def test_html_cleaner() -> None:
    """Tests the removal of HTML tags."""
    cleaner = HtmlCleaner()
    raw = "<div>Hello <br>World</div>"
    assert cleaner.clean(raw).strip() == "Hello  World"


def test_artifact_cleaner() -> None:
    """Tests the removal of URLs and emails."""
    cleaner = ArtifactCleaner()
    raw = "Contact us at admin@gritlab.ax or visit https://nordea.com today."
    assert cleaner.clean(raw).strip() == "Contact us at   or visit   today."


def test_unicode_normalizer() -> None:
    """Tests NFKC normalization and preservation of Nordic characters."""
    cleaner = UnicodeNormalizer()
    # 'a' + combining ring above (U+030A) should normalize to 'å' (U+00E5)
    raw = "Taloissani on a\u030A" 
    cleaned = cleaner.clean(raw)
    assert cleaned == "Taloissani on å"


def test_duplication_cleaner() -> None:
    """Tests the removal of excessive whitespace."""
    cleaner = DuplicationCleaner()
    raw = "   This   has   too    much   space.   "
    assert cleaner.clean(raw) == "This has too much space."


def test_full_cleaning_pipeline() -> None:
    """Tests the composite pattern chaining all cleaners."""
    pipeline = TextCleaningPipeline([
        HtmlCleaner(),
        ArtifactCleaner(),
        UnicodeNormalizer(),
        DuplicationCleaner()
    ])
    
    raw = "<p>  Visit https://site.com/a\u030A   for info! </p>"
    # Expected: HTML gone, URL gone, Unicode normalized (å), spaces compressed.
    cleaned = pipeline.clean(raw)
    assert cleaned == "Visit for info!"


def test_cleaner_type_error_exception() -> None:
    """Tests the exception block ensuring only strings are processed."""
    cleaner = HtmlCleaner()
    with pytest.raises(TypeError, match="Expected string"):
        cleaner.clean(None)  # Passing None to trigger the exception flow