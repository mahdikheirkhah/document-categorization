"""
models/tagger.py

Context-aware, multi-language document tagging — fully model-driven via SpaCy.

For each document the SpaCy pipeline (NER + POS tagger + lemmatizer) dynamically
produces:
    * Named entities (people, organizations, locations, dates, ...), whose
      model-specific labels are normalized to one standardized scheme.
    * Salient content keywords — the most frequent lemmatized nouns / proper nouns.
      These are extracted by the model, NOT from any hard-coded vocabulary, so the
      same code works across English, Swedish, and Finnish (the lemmatizer handles
      inflection / agglutination, e.g. Finnish "pankissa" -> "pankki").

The document's *topic* comes from the trained classifier (see models/pipeline.py),
so this module deliberately keeps NO hard-coded topic keyword lists. The keyword
extraction is the deterministic, rule-based heuristic that complements the
statistical NER (per the project spec).

OOP: BaseTagger (interface + tag() template) -> SpacyNerTagger (one language);
MultilingualTagger routes by language, reusing NgramLanguageDetector.
"""

from abc import ABC, abstractmethod
from collections import Counter

import spacy
from loguru import logger

from utils import config
from utils.language_detector import NgramLanguageDetector

# All tagger constants live in the central config; re-exported here as module-level
# names for callers/tests. SpaCy label schemes (en/fi OntoNotes, sv SUC) are
# normalized to one tag scheme by config.ENTITY_LABEL_TO_TAG.
SPACY_MODELS: dict[str, str] = config.SPACY_MODELS
ENTITY_LABEL_TO_TAG: dict[str, str] = config.ENTITY_LABEL_TO_TAG
KEYWORD_POS: set[str] = config.KEYWORD_POS
MAX_KEYWORDS: int = config.MAX_KEYWORDS


class BaseTagger(ABC):
    """Abstract interface for context-aware document taggers."""

    @abstractmethod
    def _analyze(self, text: str) -> dict:
        """
        Runs the underlying model and returns its dynamic signals.

        Returns:
            dict: {'entities': list[dict], 'keywords': list[str]}.
        """
        pass

    def tag(self, text: str) -> dict:
        """
        Produces context-aware tags by merging NER entity-type tags with the
        dynamically-extracted content keywords (no hard-coded vocabulary).

        Args:
            text (str): The document to tag.

        Returns:
            dict: {entities, entity_tags, keywords, tags, language}.
        """
        try:
            if not text or not str(text).strip():
                raise ValueError("Cannot tag an empty document.")

            language = getattr(self, "language", None)
            analysis = self._analyze(str(text))
            entities = analysis["entities"]
            keywords = analysis["keywords"]
            entity_tags = sorted({entity["tag"] for entity in entities})
            tags = sorted(set(entity_tags) | set(keywords))

            if not tags:
                logger.warning("No tags produced (no entities or keywords found).")

            return {
                "entities": entities,
                "entity_tags": entity_tags,
                "keywords": keywords,
                "tags": tags,
                "language": language,
            }
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Tagging failed: {e}")
            raise


class SpacyNerTagger(BaseTagger):
    """Single-language tagger backed by a SpaCy pipeline (NER + POS + lemmatizer)."""

    def __init__(
        self, language: str, model_name: str = None, max_keywords: int = MAX_KEYWORDS
    ) -> None:
        """
        Args:
            language (str): ISO 639-1 code ('en', 'sv', 'fi').
            model_name (str, optional): SpaCy model override (defaults per language).
            max_keywords (int): Number of content keywords to extract per document.
        """
        try:
            self.language = language
            self.max_keywords = max_keywords
            self.model_name = model_name or SPACY_MODELS.get(
                language, SPACY_MODELS[config.DEFAULT_LANGUAGE]
            )
            self.nlp = spacy.load(self.model_name)
            logger.info(
                f"Initialized SpacyNerTagger '{self.model_name}' for '{language}'."
            )
        except OSError as ose:
            logger.error(
                f"SpaCy model '{self.model_name}' not found. "
                f"Run: python -m spacy download {self.model_name}. Error: {ose}"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize SpacyNerTagger: {e}")
            raise

    def _analyze(self, text: str) -> dict:
        """Processes the text once: extract NER entities + salient content keywords."""
        try:
            doc = self.nlp(str(text))

            # Deduplicate entities by surface form: keep the MAJORITY label across
            # its mentions and a mention COUNT. So "Iran" becomes one row (count=9),
            # not nine rows, and a name labelled inconsistently across mentions
            # collapses to its most common label.
            mentions: dict[str, list[str]] = {}
            for ent in doc.ents:
                mentions.setdefault(ent.text, []).append(ent.label_)
            entities = []
            for surface, labels in mentions.items():
                label = Counter(labels).most_common(1)[0][0]
                entities.append(
                    {
                        "text": surface,
                        "label": label,
                        "tag": ENTITY_LABEL_TO_TAG.get(label, "misc"),
                        "count": len(labels),
                    }
                )
            entities.sort(key=lambda entity: entity["count"], reverse=True)

            # Dynamic keywords: most frequent content-word lemmas (model-driven; the
            # lemmatizer normalizes inflection, so no per-language vocabulary needed).
            lemmas = [
                token.lemma_.lower()
                for token in doc
                if token.pos_ in KEYWORD_POS
                and not token.ent_type_  # already captured as a named entity
                and not token.is_stop
                and token.is_alpha
                and len(token) > 2
            ]
            keywords = [
                word for word, _ in Counter(lemmas).most_common(self.max_keywords)
            ]

            return {"entities": entities, "keywords": keywords}
        except Exception as e:
            logger.error(f"Analysis failed ({self.language}): {e}")
            raise


class MultilingualTagger:
    """
    Routes documents to the correct language-specific SpaCy tagger, detecting the
    language when it isn't supplied. Depends only on the ``BaseTagger`` interface,
    so new languages plug in without changing this class.
    """

    def __init__(
        self,
        languages: list[str] = None,
        detector: NgramLanguageDetector = None,
    ) -> None:
        """
        Args:
            languages (list[str], optional): Supported languages. Defaults to en/sv/fi.
            detector (NgramLanguageDetector, optional): Detector for language routing.
        """
        try:
            self.languages = languages or config.SUPPORTED_LANGUAGES
            self.detector = detector or NgramLanguageDetector(self.languages)
            self.taggers: dict[str, BaseTagger] = {}  # lazily loaded per language
            logger.info(f"Initialized MultilingualTagger for {self.languages}.")
        except Exception as e:
            logger.error(f"Failed to initialize MultilingualTagger: {e}")
            raise

    def _get_tagger(self, language: str) -> BaseTagger:
        """Lazily loads and caches the SpaCy tagger for a language (fallback to en)."""
        try:
            lang = language if language in self.languages else config.DEFAULT_LANGUAGE
            if lang not in self.taggers:
                self.taggers[lang] = SpacyNerTagger(lang)
            return self.taggers[lang]
        except Exception as e:
            logger.error(f"Failed to get tagger for '{language}': {e}")
            raise

    def tag(self, text: str, language: str = None) -> dict:
        """
        Tags a document, detecting the language first when not provided.

        Args:
            text (str): The document to tag.
            language (str, optional): Known language code; detected if omitted.

        Returns:
            dict: The tag result (see ``BaseTagger.tag``) with the language used.
        """
        try:
            if not text or not str(text).strip():
                raise ValueError("Cannot tag an empty document.")
            lang = language or self.detector.detect_language(str(text))
            result = self._get_tagger(lang).tag(str(text))
            result["language"] = lang
            return result
        except Exception as e:
            logger.error(f"MultilingualTagger.tag failed: {e}")
            raise
