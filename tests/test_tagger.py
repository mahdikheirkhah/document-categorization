import pytest

from models.tagger import BaseTagger, MultilingualTagger, ENTITY_LABEL_TO_TAG


class _FakeTagger(BaseTagger):
    """Returns canned model output so the base logic can be tested without SpaCy."""

    def __init__(
        self, entities: list[dict], keywords: list[str] = None, language: str = "en"
    ) -> None:
        self.language = language
        self._entities = entities
        self._keywords = keywords or []

    def _analyze(self, text: str) -> dict:
        return {"entities": self._entities, "keywords": self._keywords}


def test_label_normalization_map() -> None:
    """English/Finnish OntoNotes and Swedish SUC labels map to one scheme."""
    assert ENTITY_LABEL_TO_TAG["PERSON"] == "person"
    assert ENTITY_LABEL_TO_TAG["PRS"] == "person"  # Swedish SUC
    assert ENTITY_LABEL_TO_TAG["ORG"] == "organization"
    assert ENTITY_LABEL_TO_TAG["GPE"] == "location"
    assert ENTITY_LABEL_TO_TAG["LOC"] == "location"
    assert ENTITY_LABEL_TO_TAG["TME"] == "date"  # Swedish SUC time
    assert ENTITY_LABEL_TO_TAG["OBJ"] == "product"


def test_tag_merges_entities_and_keywords() -> None:
    """tag() unions normalized entity-type tags with the dynamic content keywords."""
    entities = [
        {"text": "Nordea", "label": "ORG", "tag": "organization"},
        {"text": "Stockholm", "label": "GPE", "tag": "location"},
    ]
    result = _FakeTagger(entities, keywords=["bank", "savings"]).tag(
        "Some document text."
    )
    assert result["entity_tags"] == ["location", "organization"]
    assert result["keywords"] == ["bank", "savings"]
    assert set(result["tags"]) == {"location", "organization", "bank", "savings"}
    assert result["entities"] == entities
    assert result["language"] == "en"


def test_tag_empty_raises() -> None:
    """Empty/whitespace documents raise a controlled ValueError."""
    with pytest.raises(ValueError, match="empty document"):
        _FakeTagger([]).tag("   ")


def test_multilingual_router_routes_and_tags() -> None:
    """MultilingualTagger routes by language to an injected (fake) tagger."""
    router = MultilingualTagger(languages=["en", "sv", "fi"])
    router.taggers["en"] = _FakeTagger(
        [{"text": "X", "label": "ORG", "tag": "organization"}], keywords=["team"]
    )
    result = router.tag("An english text.", language="en")
    assert result["language"] == "en"
    assert "organization" in result["tags"]
    assert "team" in result["tags"]


def test_multilingual_empty_raises() -> None:
    """The router also guards against empty input."""
    router = MultilingualTagger(languages=["en"])
    with pytest.raises(ValueError):
        router.tag("", language="en")
