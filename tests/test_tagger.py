import pytest

from models.tagger import BaseTagger, MultilingualTagger, ENTITY_LABEL_TO_TAG


class _FakeTagger(BaseTagger):
    """Returns canned entities so the base logic can be tested without SpaCy."""

    def __init__(self, entities: list[dict], language: str = "en") -> None:
        self.language = language
        self._entities = entities

    def extract_entities(self, text: str) -> list[dict]:
        return self._entities


def test_label_normalization_map() -> None:
    """English (OntoNotes) and European (PER/LOC/MISC) labels map to one scheme."""
    assert ENTITY_LABEL_TO_TAG["PERSON"] == "person"
    assert ENTITY_LABEL_TO_TAG["PER"] == "person"
    assert ENTITY_LABEL_TO_TAG["ORG"] == "organization"
    assert ENTITY_LABEL_TO_TAG["GPE"] == "location"
    assert ENTITY_LABEL_TO_TAG["LOC"] == "location"


def test_base_tagger_merges_ner_and_rules() -> None:
    """tag() unions normalized NER tags with rule-based topical tags."""
    entities = [
        {"text": "Nordea", "label": "ORG", "tag": "organization"},
        {"text": "Stockholm", "label": "GPE", "tag": "location"},
    ]
    result = _FakeTagger(entities).tag(
        "The government announced an election policy meeting in Stockholm."
    )
    assert "organization" in result["entity_tags"]
    assert "location" in result["entity_tags"]
    assert "politics" in result["topic_tags"]  # government/election/policy
    assert {"organization", "location", "politics"}.issubset(set(result["tags"]))
    assert result["entities"] == entities
    assert result["language"] == "en"


def test_base_tagger_empty_raises() -> None:
    """Empty/whitespace documents raise a controlled ValueError."""
    with pytest.raises(ValueError, match="empty document"):
        _FakeTagger([]).tag("   ")


def test_multilingual_router_routes_and_tags() -> None:
    """MultilingualTagger routes by language to an injected (fake) tagger."""
    router = MultilingualTagger(languages=["en", "sv", "fi"])
    fake = _FakeTagger([{"text": "X", "label": "ORG", "tag": "organization"}])
    router.taggers["en"] = fake  # inject to avoid loading SpaCy

    result = router.tag("An english text about a team game season.", language="en")
    assert result["language"] == "en"
    assert "organization" in result["tags"]
    assert "sports" in result["topic_tags"]  # team/game/season


def test_multilingual_empty_raises() -> None:
    """The router also guards against empty input."""
    router = MultilingualTagger(languages=["en"])
    with pytest.raises(ValueError):
        router.tag("", language="en")
