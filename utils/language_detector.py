from abc import ABC, abstractmethod
from loguru import logger
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

from utils import config

# Deterministic N-gram detection across runs (reproducibility).
DetectorFactory.seed = config.SEED


class BaseLanguageDetector(ABC):
    """
    Abstract base class defining the interface for language detection.
    Ensures polymorphism if we decide to swap algorithms (e.g., FastText) later.
    """

    @abstractmethod
    def detect_language(self, text: str) -> str:
        """
        Abstract method to detect the language of a given text.

        Args:
            text (str): The raw input document.

        Returns:
            str: The ISO 639-1 language code (e.g., 'en', 'sv', 'fi').
        """
        pass


class NgramLanguageDetector(BaseLanguageDetector):
    """
    Concrete implementation utilizing Character N-gram probabilities
    to detect document language dynamically.
    """

    def __init__(self, supported_languages: list[str] = None) -> None:
        """
        Initializes the detector with a list of supported routing languages.

        Args:
            supported_languages (list[str]): Permitted language codes. Defaults to ['en', 'sv', 'fi'].
        """
        try:
            self.supported_languages = supported_languages or config.SUPPORTED_LANGUAGES
            logger.info(
                f"Initialized NgramLanguageDetector. Supported partitions: {self.supported_languages}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize NgramLanguageDetector: {e}")
            raise

    def detect_language(self, text: str) -> str:
        """
        Predicts the language using N-gram probabilities. Includes fallback logic
        for unsupported languages and structural safety checks.

        Args:
            text (str): The raw input document.

        Returns:
            str: The detected or fallback language code.
        """
        try:
            # 1. Structural Validation (Guards against "Ghost" documents)
            if not text or not str(text).strip():
                raise ValueError(
                    "Cannot detect language of an empty or whitespace-only document."
                )

            # 2. Probability Prediction
            detected_lang = detect(str(text))

            # 3. Routing Guardrails
            if detected_lang not in self.supported_languages:
                logger.warning(
                    f"Detected unsupported language '{detected_lang}'. "
                    f"Defaulting to '{config.DEFAULT_LANGUAGE}'."
                )
                return config.DEFAULT_LANGUAGE  # fallback for unsupported text

            return detected_lang

        except LangDetectException as lde:
            # This triggers if the text has no recognizable alphabet characters (e.g., "12345!@#")
            logger.error(f"N-gram detection failed on text '{text}': {lde}")
            raise ValueError(
                f"Unrecognizable characters in text, detection failed."
            ) from lde
        except ValueError as ve:
            logger.error(f"Validation error: {ve}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during language detection: {e}")
            raise
