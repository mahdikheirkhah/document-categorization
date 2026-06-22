import pytest
import numpy as np
from sklearn.model_selection import StratifiedKFold
from models.text_classifier import BaselineClassifier, DistilBertClassifier

@pytest.fixture
def dummy_data():
    """Generates a small dataset simulating 3 distinct categories."""
    X = [
        "Stock market crashes today.", "Investments are risky.", "Financial report shows profit.",
        "The football game was amazing.", "He scored a beautiful goal.", "The team won the championship.",
        "The patient needs surgery.", "The virus is spreading rapidly.", "Doctor prescribed antibiotics."
    ]
    # 0 = Finance, 1 = Sports, 2 = Medical
    y = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    return X, y

def test_baseline_training_and_evaluation(dummy_data):
    """Tests the Baseline model and evaluates F1-Macro."""
    X, y = dummy_data
    model = BaselineClassifier(max_features=10)
    
    # StratifiedKFold implementation as required by CONTRIBUTING.md
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    f1_scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val = [y[i] for i in val_idx]
        
        model.train(X_train, y_train)
        metrics = model.evaluate(X_val, y_val)
        
        assert "f1_macro" in metrics
        f1_scores.append(metrics["f1_macro"])
        
    assert len(f1_scores) == 3

def test_distilbert_initialization():
    """Tests Deep Learning encapsulation and initialization safely without running heavy training."""
    try:
        model = DistilBertClassifier(num_classes=3)
        assert model.num_classes == 3
        assert model.model is not None
    except Exception as e:
        pytest.fail(f"DistilBertClassifier initialization failed: {e}")

def test_baseline_exception_handling():
    """Tests exception flow by passing invalid data."""
    model = BaselineClassifier()
    with pytest.raises(Exception):
        model.train(None, None)