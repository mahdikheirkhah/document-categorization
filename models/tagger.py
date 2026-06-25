"""
models/tagger.py

Context-aware, multi-language document tagging (SpaCy NER + rule-based heuristics).

Design (OOP):
    * Abstraction  -> BaseTagger defines the tagging interface + the shared
      tag() template (merge NER entities with rule-based topical tags).
    * Inheritance  -> SpacyNerTagger implements entity extraction via SpaCy.
    * Encapsulation -> models are lazily loaded and cached; helpers are private.
    * Polymorphism -> MultilingualTagger routes to any BaseTagger by language.

Classification answers "which category?"; tagging answers "what/who is in it?".
NER supplies the statistical signal (people, organizations, locations, dates),
which we normalize into standardized tags and combine with a small rule-based
topical vocabulary, per the project spec.
"""

from abc import ABC, abstractmethod

import spacy
from loguru import logger

from utils.language_detector import NgramLanguageDetector

# Full SpaCy pipelines per language (these include the NER component).
SPACY_MODELS: dict[str, str] = {
    "en": "en_core_web_sm",
    "sv": "sv_core_news_sm",
    "fi": "fi_core_news_sm",
}

# Normalize heterogeneous SpaCy entity labels into standardized, language-agnostic
# tags. English uses the OntoNotes scheme (PERSON/ORG/GPE/...); the Swedish and
# Finnish models use the simpler PER/LOC/ORG/MISC scheme — both map here.
ENTITY_LABEL_TO_TAG: dict[str, str] = {
    "PERSON": "person", "PER": "person",
    "ORG": "organization",
    "GPE": "location", "LOC": "location", "FAC": "location",
    "NORP": "group",
    "DATE": "date", "TIME": "date",
    "MONEY": "money", "PERCENT": "metric", "QUANTITY": "metric",
    "CARDINAL": "number", "ORDINAL": "number",
    "PRODUCT": "product", "EVENT": "event", "WORK_OF_ART": "work",
    "LAW": "law", "LANGUAGE": "language", "MISC": "misc",
}

# Rule-based topical tags (keyword -> topic). Merged with the NER tags so the
# system blends statistical NER with simple rule-based heuristics. Currently
# English-oriented; extend with per-language vocabularies as needed.
RULE_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "politics": ["government", "election", "president", "policy", "senate", "minister", "vote"],
    "sports": ["game", "team", "season", "player", "hockey", "baseball", "score", "league"],
    "technology": ["software", "computer", "hardware", "graphics", "windows", "encryption", "driver"],
    "science": ["space", "research", "study", "scientist", "experiment", "orbit", "medicine"],
    "religion": ["god", "church", "faith", "belief", "christian", "atheism", "bible"],
    "finance": ["market", "price", "sale", "money", "buy", "sell", "cost"],
}


class BaseTagger(ABC):
    """Abstract interface for context-aware document taggers."""

    @abstractmethod
    def extract_entities(self, text: str) -> list[dict]:
        """
        Extracts named entities from the text.

        Returns:
            list[dict]: One dict per entity: {'text', 'label', 'tag'} where 'tag'
            is the normalized (standardized) category.
        """
        pass

    def _rule_based_topics(self, text: str) -> list[str]:
        """
        Derives topical tags from a keyword vocabulary (rule-based heuristic).

        Args:
            text (str): The document text.

        Returns:
            list[str]: Sorted, unique topical tags whose keywords appear in the text.
        """
        try:
            lowered = text.lower()
            topics = [
                topic
                for topic, keywords in RULE_TOPIC_KEYWORDS.items()
                if any(keyword in lowered for keyword in keywords)
            ]
            return sorted(set(topics))
        except Exception as e:
            logger.error(f"Rule-based topic tagging failed: {e}")
            raise

    def tag(self, text: str) -> dict:
        """
        Produces context-aware tags by merging NER (statistical) with rule-based
        topical tags. Template method shared by all concrete taggers.

        Args:
            text (str): The document to tag.

        Returns:
            dict: {entities, entity_tags, topic_tags, tags, language}.
        """
        try:
            if not text or not str(text).strip():
                raise ValueError("Cannot tag an empty document.")

            entities = self.extract_entities(str(text))
            entity_tags = sorted({entity["tag"] for entity in entities})
            topic_tags = self._rule_based_topics(str(text))
            all_tags = sorted(set(entity_tags) | set(topic_tags))

            if not all_tags:
                logger.warning("No tags produced (no entities or keywords matched).")

            return {
                "entities": entities,
                "entity_tags": entity_tags,
                "topic_tags": topic_tags,
                "tags": all_tags,
                "language": getattr(self, "language", None),
            }
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Tagging failed: {e}")
            raise


class SpacyNerTagger(BaseTagger):
    """Single-language tagger backed by a SpaCy NER pipeline."""

    def __init__(self, language: str, model_name: str = None) -> None:
        """
        Args:
            language (str): ISO 639-1 code ('en', 'sv', 'fi').
            model_name (str, optional): SpaCy model override. Defaults to the
                mapping in ``SPACY_MODELS``.
        """
        try:
            self.language = language
            self.model_name = model_name or SPACY_MODELS.get(language, SPACY_MODELS["en"])
            # Keep the NER component (the tokenizer pipeline excludes it; we need it).
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Initialized SpacyNerTagger '{self.model_name}' for '{language}'.")
        except OSError as ose:
            logger.error(
                f"SpaCy model '{self.model_name}' not found. "
                f"Run: python -m spacy download {self.model_name}. Error: {ose}"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize SpacyNerTagger: {e}")
            raise

    def extract_entities(self, text: str) -> list[dict]:
        """Runs SpaCy NER and normalizes the entity labels into standardized tags."""
        try:
            document = self.nlp(str(text))
            return [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "tag": ENTITY_LABEL_TO_TAG.get(ent.label_, "misc"),
                }
                for ent in document.ents
            ]
        except Exception as e:
            logger.error(f"Entity extraction failed ({self.language}): {e}")
            raise


class MultilingualTagger:
    """
    Routes documents to the correct language-specific SpaCy tagger, detecting the
    language when it isn't supplied (inference-time routing). Depends only on the
    ``BaseTagger`` interface, so new languages plug in without changing this class.
    """

    def __init__(
        self,
        languages: list[str] = None,
        detector: NgramLanguageDetector = None,
    ) -> None:
        """
        Args:
            languages (list[str], optional): Supported languages. Defaults to en/sv/fi.
            detector (NgramLanguageDetector, optional): Language detector. Defaults
                to a fresh ``NgramLanguageDetector`` over ``languages``.
        """
        try:
            self.languages = languages or ["en", "sv", "fi"]
            self.detector = detector or NgramLanguageDetector(self.languages)
            self.taggers: dict[str, BaseTagger] = {}  # lazily loaded per language
            logger.info(f"Initialized MultilingualTagger for {self.languages}.")
        except Exception as e:
            logger.error(f"Failed to initialize MultilingualTagger: {e}")
            raise

    def _get_tagger(self, language: str) -> BaseTagger:
        """Lazily loads and caches the SpaCy tagger for a language (fallback to en)."""
        try:
            lang = language if language in self.languages else "en"
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
