import re
import unicodedata
from abc import ABC, abstractmethod
from loguru import logger

from utils import config

# ==========================================
# REGEX CONSTANTS (Clean Architecture)
# ==========================================
REGEX_HTML_TAGS = re.compile(r"<[^>]+>")
REGEX_URLS = re.compile(r"http[s]?://\S+|www\.\S+")
REGEX_EMAILS = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
REGEX_MULTIPLE_SPACES = re.compile(r"\s{2,}")

# Newsgroup / email structural noise (20 Newsgroups headers, quotes, signatures).
REGEX_NEWS_HEADER = re.compile(
    r"^(?:From|Subject|Date|Organization|Lines|Newsgroups|Message-ID|References|"
    r"Sender|Reply-To|NNTP-Posting-Host|Distribution|Path|Xref|In-Reply-To|"
    r"Followup-To|Keywords|Summary|Expires|Originator|Article-I\.D\.|X-[\w-]+):.*$",
    re.MULTILINE,
)
REGEX_QUOTED_LINE = re.compile(r"^\s*(?:>|\||[A-Za-z]{1,4}>|:\s*>?).*$", re.MULTILINE)
REGEX_ATTRIBUTION = re.compile(
    r"^(?:In article\b.*|.*\b(?:writes|wrote):)\s*$", re.MULTILINE
)
REGEX_SIGNATURE = re.compile(r"^-- ?$.*", re.MULTILINE | re.DOTALL)


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
                logger.debug(
                    f"Unicode Normalization: Preserved {non_ascii_count} non-ASCII characters."
                )

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
        logger.info(
            f"Initialized TextCleaningPipeline with {len(self.cleaners)} stages."
        )

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


# utils/text_cleaning.py (Add these below your existing classes)


class FinnishMorphologyCleaner(BaseTextCleaner):
    """
    Sanitizes Finnish text by preserving agglutinative suffixes.
    Strictly avoids aggressive stemming that would destroy word roots.
    """

    def clean(self, text: str) -> str:
        # Finnish relies on specific vowel harmony; avoid aggressive character stripping
        # We only clean structural noise, never morphological suffixes
        return text.strip()


class SwedishCompoundCleaner(BaseTextCleaner):
    """
    Protects Swedish diacritics (å, ä, ö) and preserves compound structure
    by preventing over-splitting.
    """

    def clean(self, text: str) -> str:
        # Ensure we do not decompose combined diacritics into base chars + rings
        return unicodedata.normalize("NFC", text)


class NewsgroupNoiseCleaner(BaseTextCleaner):
    """
    Removes 20 Newsgroups structural noise that leaks category labels and dilutes
    signal: email/news headers, quoted reply lines, attribution lines, and the
    trailing signature block. Equivalent to scikit-learn's
    ``remove=('headers', 'footers', 'quotes')``.
    """

    def clean(self, text: str) -> str:
        try:
            if not isinstance(text, str):
                raise TypeError(f"Expected string, got {type(text)}")
            text = REGEX_SIGNATURE.sub(" ", text)
            text = REGEX_NEWS_HEADER.sub(" ", text)
            text = REGEX_ATTRIBUTION.sub(" ", text)
            text = REGEX_QUOTED_LINE.sub(" ", text)
            return text
        except Exception as e:
            logger.error(f"NewsgroupNoiseCleaner failed: {e}")
            raise


class LanguageSpecificSanitizer(BaseTextCleaner):
    """
    Polymorphic sanitizer that applies one shared, language-agnostic cleaning
    sequence to every document, then a light language-specific tweak.

    The pipeline is deliberately LIGHT (no lowercasing, stopword removal, or
    stemming) so it suits transformer models, which rely on casing and context.
    Every language gets: HTML strip -> newsgroup-noise strip -> URL/email strip
    -> unicode normalization -> (language tweak) -> whitespace normalization.
    """

    def __init__(self, lang_code: str) -> None:
        # Applied to every language.
        common_cleaners = [
            HtmlCleaner(),
            NewsgroupNoiseCleaner(),
            ArtifactCleaner(),
            UnicodeNormalizer(),
        ]
        # Light, language-specific adjustments.
        language_cleaners = {
            "en": [],
            "sv": [SwedishCompoundCleaner()],
            "fi": [FinnishMorphologyCleaner()],
        }
        self.lang_code = (
            lang_code if lang_code in language_cleaners else config.DEFAULT_LANGUAGE
        )
        # Whitespace normalization always runs last.
        self.cleaners = (
            common_cleaners + language_cleaners[self.lang_code] + [DuplicationCleaner()]
        )
        logger.info(
            f"Initialized LanguageSpecificSanitizer for '{self.lang_code}' "
            f"with {len(self.cleaners)} stages."
        )

    def clean(self, text: str) -> str:
        try:
            if not isinstance(text, str):
                raise TypeError(f"Expected string, got {type(text)}")
            for cleaner in self.cleaners:
                text = cleaner.clean(text)
            return text
        except Exception as e:
            logger.error(f"LanguageSpecificSanitizer failed: {e}")
            raise


# utils/text_cleaning.py


class CleaningObservabilityDecorator(BaseTextCleaner):
    """
    Decorator that logs the impact of the cleaning process (e.g., character count changes).
    """

    def __init__(self, cleaner: BaseTextCleaner) -> None:
        self.cleaner = cleaner

    def clean(self, text: str) -> str:
        initial_len = len(text)
        non_ascii = sum(1 for c in text if ord(c) > 127)

        result = self.cleaner.clean(text)

        logger.debug(
            f"Cleaner {type(self.cleaner).__name__}: "
            f"Removed {initial_len - len(result)} chars. "
            f"Initial Non-ASCII count: {non_ascii}"
        )
        return result
