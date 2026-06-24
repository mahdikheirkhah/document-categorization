from huggingface_hub.inference._generated.types import zero_shot_image_classification
from huggingface_hub.inference._generated.types import zero_shot_image_classification
import os
# CRITICAL: Force TensorFlow to use Legacy Keras before importing Hugging Face
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from abc import ABC, abstractmethod
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from utils.tokenization import TokenizerFactory
# Regulatory Compliance: Set global seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class BaseTextClassifier(ABC):
    """
    Abstract interface for text classifiers ensuring polymorphic behavior.
    """

    @abstractmethod
    def train(self, X_train: list[str], y_train: list[int], X_val: list[str], y_val: list[int]) -> None:
        pass

    @abstractmethod
    def predict(self, X: list[str]) -> np.ndarray:
        pass

    def evaluate(self, X: list[str], y: list[int]) -> dict[str, float]:
        """
        Standardized evaluation method computing Accuracy, F1-Macro, and AUC.
        """
        preds = self.predict(X)
        
        # Determine if predictions are probabilities (Deep Learning) or hard classes (Baseline)
        if len(preds.shape) > 1 and preds.shape[1] > 1:
            y_pred_classes = np.argmax(preds, axis=1)
            y_pred_probs = preds
        else:
            y_pred_classes = preds
            y_pred_probs = None # Baseline might need a different method for probas

        accuracy = accuracy_score(y, y_pred_classes)
        f1_macro = f1_score(y, y_pred_classes, average="macro")

        metrics = {
            "accuracy": accuracy,
            "f1_macro": f1_macro
        }

        # Calculate AUC if probabilities are available and multiclass
        if y_pred_probs is not None:
            lb = LabelBinarizer()
            y_bin = lb.fit_transform(y)
            try:
                auc = roc_auc_score(y_bin, y_pred_probs, average="macro", multi_class="ovr")
                metrics["auc"] = auc
            except ValueError as e:
                logger.warning(f"Could not calculate AUC (possibly single class in batch): {e}")

        logger.info(f"Evaluation Metrics -> Accuracy: {accuracy:.4f} | F1-Macro: {f1_macro:.4f}")
        return metrics


class BaselineClassifier(BaseTextClassifier):
    """
    TF-IDF + Logistic Regression Baseline. 
    Highly interpretable, fast, but lacks semantic understanding.
    """

    def __init__(self, max_features: int = 10000) -> None:
        self.tokenizer_factory = TokenizerFactory()
        self.eng_tokenizer = self.tokenizer_factory.get_tokenizer("en")
        
        # Inject our custom tokenizer into TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            tokenizer=self.eng_tokenizer.tokenize,
            token_pattern=None # Required by sklearn when using a custom tokenizer
        )
        self.model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        logger.info("Initialized BaselineClassifier with custom TokenizerFactory.")

    def train(self, X_train: list[str], y_train: list[int], X_val: list[str] = None, y_val: list[int] = None) -> None:
        try:
            logger.info("Training Baseline Model...")
            X_train_vec = self.vectorizer.fit_transform(X_train)
            self.model.fit(X_train_vec, y_train)
            logger.info("Baseline training complete.")
            
            if X_val and y_val:
                logger.info("Evaluating Baseline on Validation Set:")
                self.evaluate(X_val, y_val)
        except Exception as e:
            logger.error(f"Baseline training failed: {e}")
            raise

    def predict(self, X: list[str]) -> np.ndarray:
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)

    def predict_proba(self, X: list[str]) -> np.ndarray:
        X_vec = self.vectorizer.transform(X)
        return self.model.predict_proba(X_vec)


class DistilBertClassifier(BaseTextClassifier):
    """
    Deep Learning architecture using transfer learning (DistilBERT).
    Optimized for high-throughput contextual categorization.
    """

    def __init__(self, num_classes: int, model_name: str = "distilbert-base-multilingual-cased", max_length: int = 256) -> None:
        try:
            self.max_length = max_length
            self.num_classes = num_classes
            self.model_name = model_name
            self.checkpoint_dir = "../models/checkpoints/"
            
            os.makedirs(self.checkpoint_dir, exist_ok=True)

            logger.info(f"Loading tokenizer and pre-trained weights for {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = TFAutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=num_classes, 
                use_safetensors=False,
                trust_remote_code=False
            )
            
            # Using Adam optimizer with a learning rate of 3e-5 as required
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=3e-5)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            
            self.model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
            logger.info("DistilBertClassifier initialized and compiled.")
        except Exception as e:
            logger.error(f"Failed to initialize DistilBertClassifier: {e}")
            raise

    def _prepare_tf_dataset(self, texts: list[str], labels: list[int] = None, batch_size: int = 16) -> tf.data.Dataset:
        """Encapsulated helper to tokenize and format data for TensorFlow."""
        # Changed padding to "max_length" to fix Apple Silicon Metal GPU recompilation bug
        encodings = self.tokenizer(texts, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="tf")
        
        if labels is not None:
            dataset = tf.data.Dataset.from_tensor_slices((dict(encodings), labels))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((dict(encodings)))
            
        return dataset.batch(batch_size)

    def train(self, X_train: list[str], y_train: list[int], X_val: list[str], y_val: list[int], epochs: int = 5) -> None:
        try:
            logger.info("Preparing TensorFlow datasets...")
            train_ds = self._prepare_tf_dataset(X_train, y_train)
            val_ds = self._prepare_tf_dataset(X_val, y_val)

            # 1. Checkpoint Callback (.h5 equivalent weights)
            checkpoint_path = os.path.join(self.checkpoint_dir, "text_classifier_best.weights.h5")
            checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path, save_best_only=True, save_weights_only=True, monitor="val_loss", mode="min"
            )

            # 2. History Logging Callback (.csv)
            csv_path = os.path.join(self.checkpoint_dir, "training_history.csv")
            csv_cb = tf.keras.callbacks.CSVLogger(csv_path)

            logger.info(f"Starting Fine-Tuning for {epochs} epochs...")
            self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=[checkpoint_cb, csv_cb]
            )
            
            # 3. Save configuration metadata
            config_path = os.path.join(self.checkpoint_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"model_name": self.model_name, "num_classes": self.num_classes, "max_length": self.max_length}, f)
                
            logger.info(f"Training complete. Artifacts saved to {self.checkpoint_dir}")
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            raise

    def predict(self, X: list[str]) -> np.ndarray:
        try:
            test_ds = self._prepare_tf_dataset(X, batch_size=16)
            logits = self.model.predict(test_ds).logits
            # Convert logits to probabilities via Softmax
            return tf.nn.softmax(logits, axis=-1).numpy()
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise