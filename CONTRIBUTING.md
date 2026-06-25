# Contributing to the Document Categorization & Tagging Project

Thank you for contributing to our intelligent **document categorization and tagging** system!
This project applies NLP and transfer learning (DistilBERT) to classify documents and tag them
with SpaCy/NER across **three languages — English, Swedish, and Finnish**. To keep the codebase
clean, scalable, and reproducible, every contributor must follow the guidelines below.

## 1. Development Workflow (Branching & CI/CD)

* **Branching Strategy:** Never commit or push directly to the `main` branch. Always create a
  dedicated branch named after the work and (where relevant) the issue, e.g.
  `feature/3-transfer-learning-classifier`, `feature/data-fetcher`, `fix/finnish-tokenization`,
  `experiment/distilbert-vs-baseline`.
* **CI/CD Checks:** Your branch must pass all automated checks (Black formatting, linting, and the
  `pytest` suite) before it can be merged.
* **Merging:** Once the pipeline is green, open a Pull Request (PR) to `main` and request a review.
  Keep PRs scoped to a single issue where possible.

## 2. Dependency Management & Formatting

* **Poetry:** We use **Poetry** for dependency resolution and environment management. Install with
  `poetry install`. Core dependencies include `tensorflow`, `tf-keras`, `transformers`, `datasets`,
  `spacy`, `langdetect`, `streamlit`, `pandas`, `scikit-learn`, and `loguru`.
* **SpaCy / Translation models:** Language assets are downloaded separately, e.g.
  `python -m spacy download en_core_web_sm` (English), `sv_core_news_sm` (Swedish). Finnish
  sub-word handling and the EN→SV / EN→FI OPUS-MT translation models are pulled from Hugging Face.
* **Black Formatter:** We enforce a uniform code style (`line-length = 88`). Before committing, run:

```bash
poetry run black .
```

## 3. Architecture & Paradigm (OOP)

All code must be structured using **Object-Oriented Programming**. Encapsulate related logic in
well-defined classes and prefer composition of small, single-purpose objects. Our existing modules
are the reference pattern — `BaseTextCleaner`, `BaseTokenizer`, `BaseLanguageDetector`,
`BaseDatasetFetcher`, and `BaseTextClassifier` each define an abstract contract that concrete
classes implement.

Every contribution must demonstrate the **four OOP principles**:

* **Abstraction (Interfaces):** Define an `abc.ABC` base class with `@abstractmethod`s that declares
  *what* a component does, hiding *how* it does it (e.g. `BaseDatasetFetcher.fetch()`).
* **Inheritance:** Concrete classes inherit the contract and share reusable logic through base
  classes — including multi-level hierarchies (e.g.
  `BaseDatasetFetcher → BaseTranslationFetcher → SwedishTranslationFetcher`).
* **Encapsulation:** Keep internal state and helpers private (prefix with `_`), expose only a small
  public surface, and validate inputs inside the object rather than leaking complexity to callers.
* **Polymorphism:** Callers should depend on the base type, not the concrete class. Orchestrators
  (e.g. `MultilingualCorpusFetcher`) iterate over `list[BaseDatasetFetcher]` and call `fetch()`
  without knowing the concrete subtype.

When adding a new capability, first ask: *"Which existing abstract base does this extend?"* If none
fits, define a new base class with abstract methods before writing concrete implementations.

## 4. Coding Standards & Naming Conventions

* **Naming:** Use clear, descriptive names following standard Python conventions — `snake_case` for
  variables/functions (`^[a-z_][a-z0-9_]*$`) and `PascalCase` for classes (`^[A-Z][a-zA-Z0-9]*$`).
  Use consistent, shared column names across the pipeline (e.g. `text`, `label`, `label_text`,
  `language`) so language-specific frames can be concatenated without bespoke glue code.
* **Logging over Printing:** **Never use `print()`.** Use **Loguru** to record pipeline flow,
  evaluation metrics, and errors:

```python
from loguru import logger

logger.info("Loaded 11,314 English documents from 20 Newsgroups.")
logger.warning("Detected unsupported language 'es'. Defaulting to 'en'.")
logger.error("SpaCy model 'sv_core_news_sm' not found. Run 'python -m spacy download'.")
```

## 5. Function & Method Design

* **Single Responsibility Principle:** Each function/method serves **exactly one purpose**. Break
  monolithic functions into smaller, reusable pieces (load → clean → detect language → tokenize →
  classify → tag).
* **Type Hinting:** Explicitly declare argument and return types for every function and method.

```python
import pandas as pd

def fetch(self) -> pd.DataFrame:
    ...
```

* **Documentation (Docstrings):** Every function and method must include a docstring explaining:
  1. The goal and behavior of the function.
  2. The types and descriptions of the input parameters.
  3. The type and description of the return value.

## 6. Error Handling

* **Mandatory Try/Except:** Wrap the logic of **each method** in `try`/`except`, log the failure with
  Loguru, and re-raise (do not silently swallow exceptions).
* **Granular, Specific Exceptions:** Catch the most specific exception relevant to the block, e.g.
  `OSError` for a missing SpaCy model, `LangDetectException` for undetectable text, `KeyError` for a
  missing DataFrame column, and `ValueError` for empty/whitespace-only ("ghost") documents or NaN
  input during tokenization.

## 7. Model Integrity & Reproducibility

* **No Data Leakage:** Split into train/validation/test **before** fitting any vectorizer or model.
  Fit the `TfidfVectorizer` and any encoders **only on the training split**, then transform the
  others. Strip 20 Newsgroups headers/footers/quoted reply text so the model learns content, not
  metadata.
* **Imbalance-Aware Evaluation:** Plain accuracy can be misleading on skewed category distributions.
  Always report **macro-F1** alongside accuracy, use `class_weight="balanced"` where appropriate, and
  use `StratifiedKFold` / stratified splits to preserve category ratios.
* **Multilingual Parity:** Labels are language-agnostic — the Swedish and Finnish sets are produced by
  translating the English source, so all languages **share one identical label space**. Always report
  **per-language** accuracy (target ≥ 80% for `en`, `sv`, and `fi`), not just a global number.
* **Reproducibility:** Set and document `random_state` / seeds for every stochastic process
  (`np.random.seed`, `tf.random.set_seed`, `DetectorFactory.seed`, dataset sampling, and splits).

## 8. Multi-Language Data Integrity

* **Unicode Safety:** Parse as UTF-8 and normalize (NFC/NFKC) so Swedish/Finnish characters
  (`å`, `ä`, `ö`) are preserved, never decomposed or stripped. Log non-ASCII counts for observability.
* **Language-Specific Tokenization:** Route by detected language — word-level SpaCy tokenization for
  English/Swedish, sub-word (WordPiece) tokenization for agglutinative Finnish. Never apply a single
  uniform tokenizer across all languages.

## 9. Testing

* **Test-Driven Collaboration:** If you write a new function, class, or method, you must add the
  corresponding test under `tests/`.
* **Flow Coverage:** Tests must cover the main path **and** the `except` blocks (trigger known errors,
  e.g. a missing column, a NaN token, an empty document, an unsupported language fallback). Inject
  fakes/mocks for heavy models (translation, DistilBERT) so the suite runs fast and offline.
* **Classifier Checks:** Validate F1-macro and accuracy via `StratifiedKFold` splits, and assert that
  the deep model meaningfully outperforms the TF-IDF + Logistic Regression baseline.
