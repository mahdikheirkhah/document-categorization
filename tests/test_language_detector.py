import pytest
from utils.language_detector import NgramLanguageDetector

@pytest.fixture
def detector() -> NgramLanguageDetector:
    """Pytest fixture to initialize the detector once for all tests."""
    return NgramLanguageDetector()

def test_detect_english(detector: NgramLanguageDetector) -> None:
    """Tests if English is routed correctly."""
    text = "This is a standard English document about machine learning."
    assert detector.detect_language(text) == "en"

def test_detect_swedish(detector: NgramLanguageDetector) -> None:
    """Tests if Swedish (with compound words and special characters) is routed correctly."""
    text = "Detta är ett svenskt dokument om mjukvaruutveckling och datavetenskap."
    assert detector.detect_language(text) == "sv"

def test_detect_finnish(detector: NgramLanguageDetector) -> None:
    """Tests if Finnish (highly agglutinative) is routed correctly."""
    text = "Tämä on suomenkielinen asiakirja, joka käsittelee tekoälyä ja koneoppimista."
    assert detector.detect_language(text) == "fi"

def test_unsupported_language_fallback(detector: NgramLanguageDetector) -> None:
    """Tests if an unsupported language (Spanish) falls back to English safely."""
    text = "Este es un documento en español sobre bases de datos."
    # Should warn and default to 'en'
    assert detector.detect_language(text) == "en"

def test_empty_string_exception(detector: NgramLanguageDetector) -> None:
    """Tests the exception block for whitespace/ghost documents."""
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        detector.detect_language("   \n  ")

def test_unrecognizable_characters_exception(detector: NgramLanguageDetector) -> None:
    """Tests the LangDetectException block for inputs with no alphabet characters."""
    with pytest.raises(ValueError, match="Unrecognizable characters"):
        detector.detect_language("1234567890 !@#$%^&*()")