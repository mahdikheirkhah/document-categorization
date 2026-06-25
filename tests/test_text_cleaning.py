import pytest
from utils.text_cleaning import (
    HtmlCleaner,
    ArtifactCleaner,
    UnicodeNormalizer,
    DuplicationCleaner,
    TextCleaningPipeline,
    NewsgroupNoiseCleaner,
    LanguageSpecificSanitizer,
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


def test_newsgroup_noise_cleaner() -> None:
    """Strips email/news headers, quoted reply lines, attribution, and signature."""
    cleaner = NewsgroupNoiseCleaner()
    raw = (
        "From: john@example.com\n"
        "Subject: Re: hockey\n"
        "\n"
        "In article <123> someone writes:\n"
        "> I think the team is great\n"
        ": > and they will win it all\n"
        "I disagree, the season is long.\n"
        "-- \n"
        "John Doe, my signature here"
    )
    cleaned = cleaner.clean(raw)
    assert "From:" not in cleaned
    assert "Subject:" not in cleaned
    assert "writes:" not in cleaned
    assert ">" not in cleaned
    assert "my signature here" not in cleaned
    assert "I disagree, the season is long." in cleaned


def test_language_specific_sanitizer_unified_pipeline() -> None:
    """Every language now gets HTML + newsgroup + URL/email + unicode + whitespace."""
    raw = "From: a@b.com\n<p>Visit https://x.com</p>\n> quoted line\nReal Swedish content åäö."
    cleaned = LanguageSpecificSanitizer("sv").clean(raw)
    assert "From:" not in cleaned          # header removed
    assert "<p>" not in cleaned            # html removed
    assert "https://x.com" not in cleaned  # url removed
    assert ">" not in cleaned              # quote removed
    assert "åäö" in cleaned                # Swedish diacritics preserved
    assert "Real Swedish content" in cleaned


def test_language_specific_sanitizer_unknown_lang_falls_back_to_en() -> None:
    """An unsupported language code falls back to the English strategy."""
    sanitizer = LanguageSpecificSanitizer("xx")
    assert sanitizer.lang_code == "en"
    # Still cleans without raising.
    assert sanitizer.clean("Plain text here.") == "Plain text here."