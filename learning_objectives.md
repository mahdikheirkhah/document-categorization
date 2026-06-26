# Learning Objectives

This document records the NLP and ML concepts studied while building the
multi-language document categorization and tagging system, and — honestly — which
ones are actually implemented in this codebase versus understood in theory.

**Implementation legend**
- ✅ **Implemented** — built and used in the pipeline.
- ⚠️ **Partial** — partially implemented, or implemented differently than the pure theory.
- ❌ **Concept only** — understood, but not implemented in this project (with the reason).

---

## Table of Contents

- [1. Project Setup, Dataset Selection, and EDA](#1-project-setup-dataset-selection-and-eda)
  - [1.1 High-Volume, Multi-Language Data Ingestion](#11-high-volume-multi-language-data-ingestion)
  - [1.2 Statistical Challenges in NLP](#12-statistical-challenges-in-nlp)
  - [1.3 OOP Normalization Across Linguistic Families](#13-oop-normalization-across-linguistic-families)
  - [1.4 Ephemeral Embeddings vs Vector Databases](#14-ephemeral-embeddings-vs-vector-databases)
  - [1.5 Self-Attention Beyond Bag-of-Words](#15-self-attention-beyond-bag-of-words)
  - [1.6 Sub-word vs Token-Level Processing](#16-sub-word-vs-token-level-processing)
- [2. Multi-Language Preprocessing Pipeline](#2-multi-language-preprocessing-pipeline)
  - [2.1 Regex and Unicode Normalization](#21-regex-and-unicode-normalization)
  - [2.2 Automatic Language Detection](#22-automatic-language-detection)
  - [2.3 Language-Specific Tokenization](#23-language-specific-tokenization)
- [3. Transfer-Learning Classification Model](#3-transfer-learning-classification-model)
  - [3.1 Baseline ML vs Deep Learning](#31-baseline-ml-vs-deep-learning)
  - [3.2 Transfer Learning and Fine-Tuning](#32-transfer-learning-and-fine-tuning)
  - [3.3 F1-Macro vs the Accuracy Trap](#33-f1-macro-vs-the-accuracy-trap)
- [4. Information Extraction and Context-Aware Tagging](#4-information-extraction-and-context-aware-tagging)
  - [4.1 Classification vs Tagging](#41-classification-vs-tagging)
  - [4.2 Named Entity Recognition (NER)](#42-named-entity-recognition-ner)
  - [4.3 Context-Aware Logic: ML + Heuristics](#43-context-aware-logic-ml--heuristics)
- [5. Real-Time Inference and Optimization](#5-real-time-inference-and-optimization)
  - [5.1 Model Optimization: Pruning and Quantization](#51-model-optimization-pruning-and-quantization)
  - [5.2 Latency vs Accuracy Tradeoffs](#52-latency-vs-accuracy-tradeoffs)
  - [5.3 Batched and Asynchronous Pipelines](#53-batched-and-asynchronous-pipelines)
- [6. Visualization and Documentation](#6-visualization-and-documentation)
  - [6.1 Translating ML Metrics into Dashboards](#61-translating-ml-metrics-into-dashboards)
  - [6.2 Monitoring and Model Drift](#62-monitoring-and-model-drift)
  - [6.3 Documenting the "Why"](#63-documenting-the-why)

---

## 1. Project Setup, Dataset Selection, and EDA

### 1.1 High-Volume, Multi-Language Data Ingestion
Large corpora can exceed memory; chunking/streaming keeps a memory-safe flow from
source into the pipeline so the system scales with dataset size.

> **In this project: ❌ Concept only.** The MASSIVE corpus (34,542 native docs across
> en/sv/fi) loads fully into a DataFrame via `datasets.load_dataset(...).to_pandas()`
> in [utils/data_loader.py](utils/data_loader.py). Chunking/streaming wasn't needed at
> this scale and isn't implemented.

### 1.2 Statistical Challenges in NLP
Text datasets carry anomalies that hurt models: **class imbalance** (bias to the
majority class), **document-length skew** (uneven feature spaces), and
**NLP-specific "missing values"** (empty strings, whitespace, encoding corruption).

> **In this project: ⚠️ Partial.** The EDA notebook detects all three (imbalance
> ratio, IQR length outliers, and an NLP missing-value taxonomy). Mitigations
> actually applied: `class_weight="balanced"` in the baseline, length standardization
> via DistilBERT `max_length` truncation/padding, and **dropping** empty/micro docs.
> Not done: over/under-sampling, class weights for DistilBERT, or semantic imputation
> (we drop irrecoverable docs rather than infer their content).

### 1.3 OOP Normalization Across Linguistic Families
English/Swedish (Germanic) have clear word boundaries; Finnish (Finno-Ugric) is
agglutinative. An OOP design with abstract base classes encapsulates per-language
logic via inheritance for reuse and clean architecture.

> **In this project: ✅ Implemented.** Abstract bases + inheritance throughout —
> `BaseDataLoader`, `BaseTextCleaner`, `BaseTokenizer`, `BaseTagger`,
> `BaseTextClassifier`. (Note: for the transformer we keep cleaning *light* — no
> stemming; lemmatization is used only in the tagger's keyword extraction.)

### 1.4 Ephemeral Embeddings vs Vector Databases
Vector databases suit retrieval/RAG/similarity search. A direct classifier doesn't
need to persist embeddings — DistilBERT computes them internally, passes them
through its head to a softmax distribution, then discards them.

> **In this project: ✅ design / ⚠️ storage.** No vector DB; embeddings are
> ephemeral. The final structured output is written to CSV/JSON in
> [reports/](reports/) and shown in the dashboard — not a relational database.

### 1.5 Self-Attention Beyond Bag-of-Words
TF-IDF/Bag-of-Words ignore word order and meaning. Transformer self-attention weighs
every word against every other, capturing context (e.g. "bank" by a river vs money).

> **In this project: ✅ Implemented.** DistilBERT (self-attention, transfer learning)
> vs the TF-IDF baseline in [models/text_classifier.py](models/text_classifier.py).

### 1.6 Sub-word vs Token-Level Processing
Sub-word (WordPiece) tokenization splits complex/unknown words into chunks
(`unhappiness` → `un`, `##happi`, `##ness`), which suits agglutinative Finnish and
avoids vocabulary explosion.

> **In this project: ✅ Implemented (with a correction).** WordPiece is used for
> classification (DistilBERT tokenizer) and for Finnish in
> [utils/tokenization.py](utils/tokenization.py). *Correction:* NER tagging uses
> SpaCy's **token-level** pipeline (tokenizer → tagger → NER), not "sentence
> tokenization."

---

## 2. Multi-Language Preprocessing Pipeline

### 2.1 Regex and Unicode Normalization
Raw text carries structural noise (HTML, URLs, headers, signatures) and encoding
variation. Regex strips noise; Unicode normalization (NFKC) standardizes characters.

> **In this project: ✅ Implemented.** Named `REGEX_*` constants, a
> `NewsgroupNoiseCleaner`, `UnicodeNormalizer` (NFKC, preserving `å ä ö`), and an
> observability decorator that logs non-ASCII counts — all in
> [utils/text_cleaning.py](utils/text_cleaning.py).

### 2.2 Automatic Language Detection
A single pipeline routes documents by detecting their language from character/sequence
probabilities, with no manual intervention.

> **In this project: ✅ Implemented.** `NgramLanguageDetector` (langdetect) in
> [utils/language_detector.py](utils/language_detector.py). It drives **inference-time**
> routing (tagger/pipeline); during training we route by the known `language` column.

### 2.3 Language-Specific Tokenization
A uniform tokenizer destroys meaning across families: English (analytic), Swedish
(protect `å ä ö` and compounds), Finnish (mandatory sub-word splitting).

> **In this project: ✅ Implemented.** `TokenizerFactory` routes en/sv to SpaCy and
> fi to WordPiece sub-word tokenization in [utils/tokenization.py](utils/tokenization.py).

---

## 3. Transfer-Learning Classification Model

### 3.1 Baseline ML vs Deep Learning
A baseline justifies the cost of deep learning. Bag-of-Words baselines capture
frequency but not order/meaning; deep models learn dense contextual embeddings.

> **In this project: ✅ Implemented.** `BaselineClassifier` (TF-IDF + Logistic
> Regression, ~85% accuracy) vs `DistilBertClassifier` (~90%); the deep model beats the
> baseline by ~5 points (meeting the ≥5% requirement).

### 3.2 Transfer Learning and Fine-Tuning
Pre-trained models avoid learning language from scratch. DistilBERT is chosen over
BERT because lower parameter count means lower inference latency (the ≥100 docs/sec bar).

> **In this project: ✅ Implemented (with a correction).** DistilBERT-multilingual is
> fine-tuned on the corpus. *Correction:* we perform **full fine-tuning** — all layers
> are updated at a small learning rate (3e-5) — **not** frozen-base/head-only
> feature-extraction (which the original draft described).

### 3.3 F1-Macro vs the Accuracy Trap
On imbalanced data, accuracy is misleading (a majority-class predictor looks good).
Macro-F1 (harmonic mean of precision and recall, averaged equally across classes)
fixes this.

> **In this project: ✅ Implemented.** `evaluate()` reports accuracy **and** macro-F1
> (and AUC); macro-F1 is the headline metric in
> [reports/performance_metrics.json](reports/performance_metrics.json).

---

## 4. Information Extraction and Context-Aware Tagging

### 4.1 Classification vs Tagging
Classification assigns one mutually-exclusive label (*what the document is*); tagging
attaches multiple non-exclusive markers (*what the document contains*). They complement
each other.

> **In this project: ✅ Implemented.** Single-label softmax classification +
> multi-tag NER/keyword tagging, combined in
> [models/pipeline.py](models/pipeline.py).

### 4.2 Named Entity Recognition (NER)
NER turns unstructured text into structured entities (persons, organizations,
locations, dates), using surrounding syntax to disambiguate meaning.

> **In this project: ✅ Implemented.** `SpacyNerTagger` in
> [models/tagger.py](models/tagger.py), with label normalization across the
> en/fi OntoNotes scheme and the Swedish SUC scheme, plus per-mention deduplication.

### 4.3 Context-Aware Logic: ML + Heuristics
Robust pipelines combine statistical predictions with deterministic logic.

> **In this project: ⚠️ Partial (and reframed).** We merge statistical NER with a
> **deterministic, model-driven keyword-extraction heuristic** (top lemmatized
> non-entity nouns) and surface a **low-confidence flag** when the top class is weak.
> *Correction:* we do **not** hardcode If/Else rules that override the classifier
> based on entities — we deliberately **removed** the hardcoded keyword vocabulary in
> favor of dynamic extraction, so "hardcoded guardrails overriding the model" is not
> what this system does.

---

## 5. Real-Time Inference and Optimization

### 5.1 Model Optimization: Pruning and Quantization
**Quantization** lowers weight precision (e.g. fp32 → int8) for faster math and less
RAM; **pruning** removes near-zero connections to create a sparse, cheaper model.

> **In this project: ✅ Implemented.** Both, via an OOP `BaseModelOptimizer` hierarchy
> in [models/optimization.py](models/optimization.py), using **core TensorFlow** rather
> than the `tensorflow_model_optimization` toolkit — tfmot's pruning needs prune-aware
> re-training and does not cleanly wrap Hugging Face's custom transformer layers, so a
> dependency-free, GPU-free approach was chosen:
> - **Magnitude pruning** (`MagnitudePruningOptimizer`) zeros the smallest 30% of
>   weights in the large 2-D kernels/embeddings (~40.6M weights) with **no accuracy
>   loss** (~0.90 retained). Pruned weights reload into the *same* architecture, so the
>   serving pipeline needs no new code.
> - **Float16 quantization** (`Float16Quantizer`): 32→16-bit weights, **541 MB → 200 MB
>   (−63%)**, accuracy unchanged.
> - **Dynamic-range int8** (`TFLiteDynamicRangeQuantizer`, TFLite): **541 MB → 137 MB
>   (−75%)** with faster CPU inference.
>
> Run with `python optimize.py`; measurements land in
> [reports/optimization_metrics.json](reports/optimization_metrics.json). The served
> variant is selectable via `SERVING_WEIGHTS_FILENAME` in
> [utils/config.py](utils/config.py).

### 5.2 Latency vs Accuracy Tradeoffs
Heavier models are more accurate but slower; the design must stay fast enough
(≥100 docs/sec) while staying accurate enough (≥85% / 0.80 macro-F1).

> **In this project: ✅ Implemented.** Addressed first by **model choice** (DistilBERT
> over BERT) and verified by the benchmark (~305 docs/sec on GPU at ~90% accuracy),
> then **further** by the pruning/quantization in [§5.1](#51-model-optimization-pruning-and-quantization)
> for CPU-bound serving (a ~4× smaller int8 model).

### 5.3 Batched and Asynchronous Pipelines
Batching exploits GPU parallelism (higher throughput at slight per-item latency);
async pipelines decouple CPU tokenization from GPU inference to avoid idle hardware.

> **In this project: ⚠️ Partial.** **Batched** inference is implemented
> (`process_batch` runs one model call per batch). **Asynchronous** (async/await)
> decoupling is **not** — the pipeline is synchronous.

---

## 6. Visualization and Documentation

### 6.1 Translating ML Metrics into Dashboards
End users can't read tensors; dashboards must show operational realities (categories,
tags, confidence, speed) so stakeholders can trust the system.

> **In this project: ✅ Implemented.** The Streamlit dashboard
> [app/real_time_dashboard.py](app/real_time_dashboard.py) shows live
> categorization + tagging, accuracy/F1/throughput, per-language accuracy, and
> category/language/tag distributions. (No time-series/daily-volume view.)

### 6.2 Monitoring and Model Drift
Production models drift as real data diverges from training; tracking confidence and
distributions signals when retraining is needed.

> **In this project: ⚠️ Partial.** Per-prediction **confidence**, a **low-confidence
> flag**, **latency**, and a **session tag-count** view exist in the dashboard. True
> drift detection (time-series confidence/distribution tracking) and hardware
> telemetry are **not** implemented.

### 6.3 Documenting the "Why"
Code comments explain *what*; architectural docs explain *why* — the challenges,
alternatives, and rationale — so the system can be safely maintained and extended.

> **In this project: ✅ Implemented.** [data.md](data.md) (data/EDA/cleaning
> rationale), [CONTRIBUTING.md](CONTRIBUTING.md) (standards), module docstrings, and
> this file document the reasoning behind the architecture.
