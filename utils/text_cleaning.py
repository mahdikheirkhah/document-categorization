import re
import unicodedata
from abc import ABC, abstractmethod
from loguru import logger

# ==========================================
# REGEX CONSTANTS (Clean Architecture)
# ==========================================
REGEX_HTML_TAGS = re.compile(r"<[^>]+>")
REGEX_URLS = re.compile(r"http[s]?://\S+|www\.\S+")
REGEX_EMAILS = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
REGEX_MULTIPLE_SPACES = re.compile(r"\s{2,}")


class BaseTextCleaner(ABC):
    """
    Abstract base class defining the interface for all text cleaning strategies.
    Ensures polymorphism across the cleaning pipeline.
    """

    @abstractmethod
    def clean(self, text: str) -> str:
        """
        Abstract method to clean the input text.

        Args:
            text (str): The raw input string.

        Returns:
            str: The cleaned string.
        """
        pass


class HtmlCleaner(BaseTextCleaner):
    """Removes HTML tags from the text."""

    def clean(self, text: str) -> str:
        try:
            if not isinstance(text, str):
                raise TypeError(f"Expected string, got {type(text)}")
            return REGEX_HTML_TAGS.sub(" ", text)
        except Exception as e:
            logger.error(f"HtmlCleaner failed: {e}")
            raise


class ArtifactCleaner(BaseTextCleaner):
    """Removes URLs, emails, and leftover formatting artifacts."""

    def clean(self, text: str) -> str:
        try:
            if not isinstance(text, str):
                raise TypeError(f"Expected string, got {type(text)}")
            text = REGEX_URLS.sub(" ", text)
            text = REGEX_EMAILS.sub(" ", text)
            return text
        except Exception as e:
            logger.error(f"ArtifactCleaner failed: {e}")
            raise


class UnicodeNormalizer(BaseTextCleaner):
    """
    Normalizes UTF-8 encodings (NFKC) and provides observability by 
    logging the presence of non-ASCII characters (vital for Swedish/Finnish).
    """

    def clean(self, text: str) -> str:
        try:
            if not isinstance(text, str):
                raise TypeError(f"Expected string, got {type(text)}")
            
            # Normalize complex unicode characters (e.g., combining accents)
            normalized_text = unicodedata.normalize("NFKC", text)
            
            # Observability: Count non-ASCII characters (like å, ä, ö)
            non_ascii_count = sum(1 for char in normalized_text if ord(char) > 127)
            if non_ascii_count > 0:
                logger.debug(f"Unicode Normalization: Preserved {non_ascii_count} non-ASCII characters.")
                
            return normalized_text
        except Exception as e:
            logger.error(f"UnicodeNormalizer failed: {e}")
            raise


class DuplicationCleaner(BaseTextCleaner):
    """Strips excessive whitespace and consecutive duplicated spaces."""

    def clean(self, text: str) -> str:
        try:
            if not isinstance(text, str):
                raise TypeError(f"Expected string, got {type(text)}")
            # Replace multiple spaces with a single space and strip edges
            return REGEX_MULTIPLE_SPACES.sub(" ", text).strip()
        except Exception as e:
            logger.error(f"DuplicationCleaner failed: {e}")
            raise


class TextCleaningPipeline(BaseTextCleaner):
    """
    Composite class that chains multiple BaseTextCleaner strategies together.
    """

    def __init__(self, cleaners: list[BaseTextCleaner]) -> None:
        """
        Initializes the pipeline with a specific sequence of cleaners.

        Args:
            cleaners (list[BaseTextCleaner]): A list of instantiated cleaner classes.
        """
        self.cleaners = cleaners
        logger.info(f"Initialized TextCleaningPipeline with {len(self.cleaners)} stages.")

    def clean(self, text: str) -> str:
        """
        Executes the cleaning strategies sequentially.
        """
        try:
            if not isinstance(text, str):
                raise TypeError("Pipeline received non-string input.")
            
            cleaned_text = text
            for cleaner in self.cleaners:
                cleaned_text = cleaner.clean(cleaned_text)
                
            return cleaned_text
        except Exception as e:
            logger.error(f"TextCleaningPipeline execution failed: {e}")
            raise