import numpy as np
import pytest

from models.pipeline import RealTimePipeline


class FakeClassifier:
    """Deterministic stand-in for DistilBertClassifier (no model load)."""

    def __init__(self, num_classes: int = 3) -> None:
        self.num_classes = num_classes

    def predict(self, texts: list[str]) -> np.ndarray:
        probs = np.zeros((len(texts), self.num_classes))
        for i, text in enumerate(texts):
            cls = len(text) % self.num_classes
            probs[i, cls] = 0.9  # confident, deterministic prediction
        return probs


class FakeTagger:
    """Stand-in tagger returning a fixed tag/entity."""

    def tag(self, text: str, language: str = None) -> dict:
        return {
            "tags": ["organization"],
            "entities": [{"text": "X", "label": "ORG", "tag": "organization"}],
        }


@pytest.fixture
def pipeline() -> RealTimePipeline:
    return RealTimePipeline(
        classifier=FakeClassifier(num_classes=3),
        tagger=FakeTagger(),
        label_map={0: "finance", 1: "sports", 2: "health"},
        corpus_path=None,
        languages=["en"],
    )


def test_process_single(pipeline: RealTimePipeline) -> None:
    result = pipeline.process("The team won the championship game.", language="en")
    assert result["language"] == "en"
    assert result["category"] in {"finance", "sports", "health"}
    assert result["category_id"] in {0, 1, 2}
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["tags"] == ["organization"]
    assert isinstance(result["entities"], list)


def test_process_batch(pipeline: RealTimePipeline) -> None:
    docs = [
        "short",
        "a much longer document about several things",
        "mid length doc here",
    ]
    out = pipeline.process_batch(docs, languages=["en", "en", "en"])
    assert len(out) == 3
    assert all("category" in r and "confidence" in r for r in out)


def test_empty_doc_raises(pipeline: RealTimePipeline) -> None:
    with pytest.raises(ValueError):
        pipeline.process("   ", language="en")


def test_label_map_numeric_fallback() -> None:
    pipe = RealTimePipeline(
        classifier=FakeClassifier(3),
        tagger=FakeTagger(),
        label_map=None,
        corpus_path=None,
        languages=["en"],
    )
    result = pipe.process("hello world document", language="en")
    assert result["category"] == str(result["category_id"])


def test_benchmark(pipeline: RealTimePipeline) -> None:
    docs = [f"document number {i} with some words" for i in range(10)]
    stats = pipeline.benchmark(docs, batch_size=4)
    assert stats["documents"] == 10
    assert stats["docs_per_sec"] > 0
