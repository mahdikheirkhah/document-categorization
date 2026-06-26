import pytest
import pandas as pd
from utils.tokenization import SpacyTokenizer, FinnishSubwordTokenizer, TokenizerFactory


@pytest.fixture(scope="module")
def factory() -> TokenizerFactory:
    """Pytest fixture to initialize the heavy models only once per test run."""
    return TokenizerFactory()


def test_english_tokenization(factory: TokenizerFactory) -> None:
    """Tests standard English word tokenization."""
    tokenizer = factory.get_tokenizer("en")
    tokens = tokenizer.tokenize("Machine learning is great.")
    assert tokens == ["Machine", "learning", "is", "great", "."]


def test_swedish_tokenization(factory: TokenizerFactory) -> None:
    """Tests Swedish tokenization, ensuring special characters are preserved."""
    tokenizer = factory.get_tokenizer("sv")
    tokens = tokenizer.tokenize("Mjukvaruutveckling på Åland.")
    assert "Mjukvaruutveckling" in tokens
    assert "Åland" in tokens


def test_finnish_subword_tokenization(factory: TokenizerFactory) -> None:
    """
    Tests Finnish sub-word tokenization.
    Verifies the WordPiece algorithm correctly splits complex words using '##'.
    """
    tokenizer = factory.get_tokenizer("fi")
    # 'taloissani' (in my houses). DistilBERT should split this agglutinative word.
    tokens = tokenizer.tokenize("taloissani")

    # Verify that sub-word artifacts ('##') are present, proving it didn't just split by space
    has_subwords = any(token.startswith("##") for token in tokens)
    assert has_subwords is True


def test_nan_handling_exceptions(factory: TokenizerFactory) -> None:
    """Tests the try/except block to ensure NaNs trigger a controlled ValueError."""
    tokenizer = factory.get_tokenizer("en")

    with pytest.raises(ValueError, match="Encountered NaN or None"):
        tokenizer.tokenize(None)

    with pytest.raises(ValueError, match="Encountered NaN or None"):
        import numpy as np

        tokenizer.tokenize(np.nan)


def test_no_data_leakage(factory: TokenizerFactory) -> None:
    """
    Tests that the tokenizer maintains strict statelessness between calls.
    Ensures no data from Document A leaks into the tokens of Document B.
    """
    tokenizer = factory.get_tokenizer("en")

    doc_a = "First document string."
    doc_b = "Second document."

    tokens_a = tokenizer.tokenize(doc_a)
    tokens_b = tokenizer.tokenize(doc_b)

    # Ensure memory addresses of the lists are entirely distinct
    assert id(tokens_a) != id(tokens_b)
    # Ensure content did not bleed over
    assert "First" not in tokens_b
    assert len(tokens_a) == 4
    assert len(tokens_b) == 3
