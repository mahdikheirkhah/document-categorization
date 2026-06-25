import os

# CRITICAL: configure the Hugging Face backend BEFORE importing transformers.
# 1) Force legacy Keras (Keras 2, via tf-keras) so the TF training APIs behave.
# 2) Keep transformers TensorFlow-ONLY: this classifier is TF/Keras (required by
#    the subject and the audit), and on macOS importing TensorFlow and PyTorch in
#    one process aborts with an OpenMP "mutex lock failed". PyTorch is used only by
#    the offline translation step, which runs in its own process.
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["USE_TORCH"] = "0"

import json

import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

from utils import config
from utils.tokenization import TokenizerFactory

# Global seeds for reproducibility (per CONTRIBUTING.md).
np.random.seed(config.SEED)
tf.random.set_seed(config.SEED)

# Artifacts always land in models/checkpoints/ (resolved centrally in config).
CHECKPOINT_DIR = config.CHECKPOINT_DIR


class BaseTextClassifier(ABC):
    """Abstract interface for text classifiers, ensuring polymorphic behavior."""

    @abstractmethod
    def train(
        self, X_train: list[str], y_train: list[int], X_val: list[str], y_val: list[int]
    ) -> None:
        """Fits the classifier on the training data (and optionally validates)."""
        pass

    @abstractmethod
    def predict(self, X: list[str]) -> np.ndarray:
        """Returns predictions for X (hard classes or class probabilities)."""
        pass

    def evaluate(self, X: list[str], y: list[int]) -> dict[str, float]:
        """
        Computes Accuracy, F1-Macro, and (when probabilities are available) AUC.

        Args:
            X (list[str]): Documents to evaluate.
            y (list[int]): True integer labels.

        Returns:
            dict[str, float]: The computed metrics.
        """
        try:
            preds = self.predict(X)

            # Probabilities (deep model) vs hard classes (baseline).
            if len(preds.shape) > 1 and preds.shape[1] > 1:
                y_pred_classes = np.argmax(preds, axis=1)
                y_pred_probs = preds
            else:
                y_pred_classes = preds
                y_pred_probs = None

            accuracy = accuracy_score(y, y_pred_classes)
            f1_macro = f1_score(y, y_pred_classes, average="macro")
            metrics = {"accuracy": accuracy, "f1_macro": f1_macro}

            # AUC only makes sense with probability outputs (multiclass OvR).
            if y_pred_probs is not None:
                y_bin = LabelBinarizer().fit_transform(y)
                try:
                    metrics["auc"] = roc_auc_score(
                        y_bin, y_pred_probs, average="macro", multi_class="ovr"
                    )
                except ValueError as ve:
                    logger.warning(f"Could not calculate AUC: {ve}")

            logger.info(
                f"Evaluation Metrics -> Accuracy: {accuracy:.4f} | F1-Macro: {f1_macro:.4f}"
            )
            return metrics
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise


class BaselineClassifier(BaseTextClassifier):
    """
    TF-IDF + Logistic Regression baseline: fast and interpretable, but lacking
    semantic understanding. Establishes the floor the deep model must beat by >=5%.
    """

    def __init__(self, max_features: int = config.BASELINE_MAX_FEATURES) -> None:
        """
        Args:
            max_features (int): Maximum vocabulary size for the TF-IDF vectorizer.
        """
        try:
            # Reuse our OOP word-level tokenizer for the vectorizer.
            self.tokenizer_factory = TokenizerFactory()
            self.eng_tokenizer = self.tokenizer_factory.get_tokenizer("en")
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                tokenizer=self.eng_tokenizer.tokenize,
                token_pattern=None,  # required by sklearn when a tokenizer is given
            )
            self.model = LogisticRegression(
                max_iter=1000, random_state=config.SEED, class_weight="balanced"
            )
            logger.info("Initialized BaselineClassifier (TF-IDF + Logistic Regression).")
        except Exception as e:
            logger.error(f"Failed to initialize BaselineClassifier: {e}")
            raise

    def train(
        self,
        X_train: list[str],
        y_train: list[int],
        X_val: list[str] = None,
        y_val: list[int] = None,
    ) -> None:
        """Fits TF-IDF + Logistic Regression, optionally evaluating on a val set."""
        try:
            logger.info("Training Baseline Model...")
            X_train_vec = self.vectorizer.fit_transform(X_train)
            self.model.fit(X_train_vec, y_train)
            logger.info("Baseline training complete.")

            if X_val is not None and y_val is not None:
                logger.info("Evaluating Baseline on Validation Set:")
                self.evaluate(X_val, y_val)
        except Exception as e:
            logger.error(f"Baseline training failed: {e}")
            raise

    def predict(self, X: list[str]) -> np.ndarray:
        """Returns hard class predictions for X."""
        try:
            return self.model.predict(self.vectorizer.transform(X))
        except Exception as e:
            logger.error(f"Baseline prediction failed: {e}")
            raise

    def predict_proba(self, X: list[str]) -> np.ndarray:
        """Returns class probability estimates for X."""
        try:
            return self.model.predict_proba(self.vectorizer.transform(X))
        except Exception as e:
            logger.error(f"Baseline probability prediction failed: {e}")
            raise


class DistilBertClassifier(BaseTextClassifier):
    """
    Transfer-learning classifier built on DistilBERT (TensorFlow/Keras). Uses the
    multilingual checkpoint so one model serves English, Swedish, and Finnish.
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = config.DISTILBERT_MODEL,
        max_length: int = config.MAX_LENGTH,
    ) -> None:
        """
        Args:
            num_classes (int): Number of target categories.
            model_name (str): Hugging Face checkpoint to fine-tune.
            max_length (int): Max sub-word sequence length (truncation/padding).
        """
        try:
            self.max_length = max_length
            self.num_classes = num_classes
            self.model_name = model_name
            self.checkpoint_dir = CHECKPOINT_DIR
            os.makedirs(self.checkpoint_dir, exist_ok=True)

            logger.info(f"Loading tokenizer and pre-trained weights for {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = TFAutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_classes, use_safetensors=False
            )

            # Learning rate sits in the required 2e-5..5e-5 fine-tuning band.
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=config.LEARNING_RATE)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            self.model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
            logger.info("DistilBertClassifier initialized and compiled.")
        except Exception as e:
            logger.error(f"Failed to initialize DistilBertClassifier: {e}")
            raise

    def _prepare_tf_dataset(
        self,
        texts: list[str],
        labels: list[int] = None,
        batch_size: int = config.TRAIN_BATCH_SIZE,
    ) -> tf.data.Dataset:
        """Tokenizes and batches texts (and labels) into a tf.data.Dataset."""
        try:
            # padding="max_length" gives fixed shapes (avoids Apple Silicon Metal recompiles).
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="tf",
            )
            if labels is not None:
                dataset = tf.data.Dataset.from_tensor_slices((dict(encodings), labels))
            else:
                dataset = tf.data.Dataset.from_tensor_slices((dict(encodings)))
            return dataset.batch(batch_size)
        except Exception as e:
            logger.error(f"Failed to prepare TF dataset: {e}")
            raise

    def train(
        self,
        X_train: list[str],
        y_train: list[int],
        X_val: list[str],
        y_val: list[int],
        epochs: int = config.EPOCHS,
    ) -> None:
        """
        Fine-tunes DistilBERT, saving the best weights, training history, and config
        to models/checkpoints/.

        Args:
            X_train (list[str]): Training documents.
            y_train (list[int]): Training labels.
            X_val (list[str]): Validation documents.
            y_val (list[int]): Validation labels.
            epochs (int): Number of fine-tuning epochs (>= 5 per the spec).
        """
        try:
            logger.info("Preparing TensorFlow datasets...")
            train_ds = self._prepare_tf_dataset(X_train, y_train)
            val_ds = self._prepare_tf_dataset(X_val, y_val)

            # 1. Best-weights checkpoint (filename matches the validation spec).
            checkpoint_path = os.path.join(self.checkpoint_dir, config.BEST_WEIGHTS_FILENAME)
            checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_best_only=True,
                save_weights_only=True,
                monitor="val_loss",
                mode="min",
            )
            # 2. Per-epoch metrics for the training-history artifact.
            csv_path = os.path.join(self.checkpoint_dir, config.HISTORY_FILENAME)
            csv_cb = tf.keras.callbacks.CSVLogger(csv_path)

            logger.info(f"Starting Fine-Tuning for {epochs} epochs...")
            self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=[checkpoint_cb, csv_cb],
            )

            # 3. Persist the configuration metadata.
            config_path = os.path.join(self.checkpoint_dir, config.MODEL_CONFIG_FILENAME)
            with open(config_path, "w") as f:
                json.dump(
                    {
                        "model_name": self.model_name,
                        "num_classes": self.num_classes,
                        "max_length": self.max_length,
                    },
                    f,
                )
            logger.info(f"Training complete. Artifacts saved to {self.checkpoint_dir}")
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            raise

    def predict(self, X: list[str], batch_size: int = config.INFERENCE_BATCH_SIZE) -> np.ndarray:
        """
        Returns softmax class probabilities for X.

        Calls the model directly in manual batches with dynamic padding — far lower
        latency than ``model.predict`` (no dataset/callback overhead) for real-time
        and small-batch inference.

        Args:
            X (list[str]): Documents to classify.
            batch_size (int): Inference batch size.

        Returns:
            np.ndarray: Shape (len(X), num_classes) probability matrix.
        """
        try:
            probabilities = []
            for start in range(0, len(X), batch_size):
                batch = X[start : start + batch_size]
                encodings = self.tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="tf",
                )
                logits = self.model(dict(encodings), training=False).logits
                probabilities.append(tf.nn.softmax(logits, axis=-1).numpy())
            if not probabilities:
                return np.empty((0, self.num_classes))
            return np.vstack(probabilities)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def load_weights(self, weights_path: str = None) -> None:
        """
        Loads fine-tuned weights into the (already-built) model for inference.

        Args:
            weights_path (str, optional): Path to the ``.h5`` weights. Defaults to
                ``<checkpoint_dir>/text_classifier_best.h5``.
        """
        try:
            path = weights_path or os.path.join(self.checkpoint_dir, config.BEST_WEIGHTS_FILENAME)
            self.model.load_weights(path)
            logger.info(f"Loaded fine-tuned weights from {path}.")
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            raise

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str = CHECKPOINT_DIR) -> "DistilBertClassifier":
        """
        Rebuilds a ready-to-predict classifier from a saved checkpoint
        (``config.json`` + ``text_classifier_best.h5``).

        Args:
            checkpoint_dir (str): Directory holding the config and weights.

        Returns:
            DistilBertClassifier: Classifier with the fine-tuned weights loaded.
        """
        try:
            with open(os.path.join(checkpoint_dir, config.MODEL_CONFIG_FILENAME)) as f:
                saved_config = json.load(f)
            classifier = cls(
                num_classes=saved_config["num_classes"],
                model_name=saved_config["model_name"],
                max_length=saved_config["max_length"],
            )
            classifier.load_weights(
                os.path.join(checkpoint_dir, config.BEST_WEIGHTS_FILENAME)
            )
            return classifier
        except Exception as e:
            logger.error(f"Failed to load classifier from checkpoint: {e}")
            raise
