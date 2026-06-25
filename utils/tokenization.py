import spacy
import pandas as pd
from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from loguru import logger

from utils import config


class BaseTokenizer(ABC):
    """
    Abstract base class defining the interface for all language-specific tokenizers.
    Ensures polymorphism and clean architecture.
    """

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """
        Abstract method to tokenize the input text.

        Args:
            text (str): The cleaned input string.

        Returns:
            list[str]: A list of extracted tokens.
        """
        pass


class SpacyTokenizer(BaseTokenizer):
    """
    Concrete implementation utilizing SpaCy for word-level tokenization.
    Used primarily for Analytic (English) and Germanic (Swedish) languages.
    """

    def __init__(self, model_name: str) -> None:
        """
        Initializes the SpaCy model.

        Args:
            model_name (str): The SpaCy model to load (e.g., 'en_core_web_sm').
        """
        try:
            # Load the model but exclude heavy pipeline components to save memory
            self.nlp = spacy.load(model_name, exclude=["ner", "parser"])
            logger.info(f"Initialized SpacyTokenizer with model '{model_name}'.")
        except OSError as e:
            logger.error(f"SpaCy model '{model_name}' not found. Did you run 'python -m spacy download'? Error: {e}")
            raise

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenizes text using SpaCy, safely handling NaNs and empty strings.
        """
        try:
            # Safe NaN Handling
            if pd.isna(text) or text is None:
                raise ValueError("Encountered NaN or None value during tokenization.")
            
            text_str = str(text).strip()
            if not text_str:
                return []

            # Encapsulated processing prevents data leakage across calls
            doc = self.nlp(text_str)
            return [token.text for token in doc if not token.is_space]
        
        except ValueError as ve:
            logger.error(f"Validation error in SpacyTokenizer: {ve}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in SpacyTokenizer: {e}")
            raise


class FinnishSubwordTokenizer(BaseTokenizer):
    """
    Concrete implementation utilizing Hugging Face WordPiece for sub-word tokenization.
    Crucial for Finno-Ugric languages to prevent vocabulary explosion from agglutination.
    """

    def __init__(self, model_name: str = config.FINNISH_SUBWORD_MODEL) -> None:
        """
        Initializes the Hugging Face sub-word tokenizer.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Initialized FinnishSubwordTokenizer with '{model_name}'.")
        except Exception as e:
            logger.error(f"Failed to load sub-word tokenizer '{model_name}': {e}")
            raise

    def tokenize(self, text: str) -> list[str]:
        """
        Performs sub-word tokenization on Finnish text.
        """
        try:
            # Safe NaN Handling
            if pd.isna(text) or text is None:
                raise ValueError("Encountered NaN or None value during tokenization.")
            
            text_str = str(text).strip()
            if not text_str:
                return []

            # Returns sub-words (e.g., 'taloissani' -> ['talo', '##issa', '##ni'])
            return self.tokenizer.tokenize(text_str)
            
        except ValueError as ve:
            logger.error(f"Validation error in FinnishSubwordTokenizer: {ve}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in FinnishSubwordTokenizer: {e}")
            raise


class TokenizerFactory:
    """
    Factory class that routes text to the correct tokenizer based on language code.
    """

    def __init__(self) -> None:
        """
        Instantiates all required tokenizers on startup.
        """
        self.tokenizers: dict[str, BaseTokenizer] = {
            "en": SpacyTokenizer(config.SPACY_MODELS["en"]),
            "sv": SpacyTokenizer(config.SPACY_MODELS["sv"]),
            "fi": FinnishSubwordTokenizer(),
        }

    def get_tokenizer(self, lang_code: str) -> BaseTokenizer:
        """
        Retrieves the appropriate tokenizer, defaulting to English.
        """
        if lang_code not in self.tokenizers:
            logger.warning(
                f"No specific tokenizer for '{lang_code}'. "
                f"Defaulting to '{config.DEFAULT_LANGUAGE}'."
            )
            return self.tokenizers[config.DEFAULT_LANGUAGE]
        return self.tokenizers[lang_code]