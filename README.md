# Document Categorization & Tagging — Multilingual (EN · SV · FI)

An intelligent, **multi-language** document categorization and tagging system. It
classifies documents into categories with a fine-tuned **DistilBERT** transformer,
extracts **named entities and context-aware tags** with SpaCy NER, serves everything
through a **real-time pipeline + Streamlit dashboard**, and ships **pruned and
quantized** model variants for fast CPU/edge inference. English, Swedish, and Finnish
are supported natively.

| Metric (validation) | Result | Target |
| --- | --- | --- |
| Classification accuracy | **90.3 %** | ≥ 85 % |
| Macro-F1 | **0.894** | ≥ 0.80 |
| Throughput | **305 docs/s** (GPU) | ≥ 100 |
| Per-language accuracy | en **92.6 %** · fi **89.3 %** · sv **88.9 %** | ≥ 80 % each |
| Lift over TF-IDF baseline | **+5.2 pts** | ≥ 5 |

---

## Table of Contents

- [1. About the Project](#1-about-the-project)
- [2. Dataset](#2-dataset)
- [3. The Pipeline (Step by Step)](#3-the-pipeline-step-by-step)
- [4. Results](#4-results)
  - [4.1 Classification performance](#41-classification-performance)
  - [4.2 Before vs after pruning + quantization](#42-before-vs-after-pruning--quantization)
  - [4.3 Why pruning/quantization matter for real-time systems](#43-why-pruningquantization-matter-for-real-time-systems)
- [5. Usage](#5-usage)
  - [5.1 Prerequisites](#51-prerequisites)
  - [5.2 Run the dashboard](#52-run-the-dashboard)
  - [5.3 (Re)train the model](#53-retrain-the-model)
  - [5.4 Optimize the model (prune + quantize)](#54-optimize-the-model-prune--quantize)
- [6. Project Structure](#6-project-structure)
- [7. Further Documentation](#7-further-documentation)

---

## 1. About the Project

The system answers two complementary questions about any incoming document:

- **What is it?** — a single-label **category** (classification), e.g. `weather`,
  `calendar`, `news`. A DistilBERT transformer fine-tuned with **transfer learning**
  provides deep, context-aware predictions that beat a TF-IDF + Logistic Regression
  baseline.
- **What does it contain?** — multiple **tags** (NER + keywords), e.g. people,
  organizations, locations, dates, and content keywords. SpaCy named-entity
  recognition runs per language and is normalized into one tag scheme.

Everything is **multilingual** (English, Swedish, Finnish), built with strict OOP
(abstract base classes + inheritance for every component), and designed for
**real-time** use: batched inference, a language-routing pipeline, and optimized
(pruned/quantized) model artifacts for CPU/edge serving.

## 2. Dataset

We use the **MASSIVE** dataset (Amazon) — a natively multilingual, *parallel* corpus
(`mteb/amazon_massive_scenario` on Hugging Face). The same utterances exist in every
language, so English, Swedish, and Finnish **share one identical label space** with no
translation step. We classify the **18 "scenario" domains** (`alarm`, `calendar`,
`weather`, `news`, `qa`, `general`, …).

- **Size:** 34,542 documents — **11,514 per language** (en / sv / fi).
- **Schema:** `text | label | label_text | language`.
- **Requirements met:** ≥ 10,000 docs ✓, ≥ 5 categories (18) ✓, ≥ 2 languages (3) ✓.

> 📄 Full details on the data source, exploratory analysis, and per-language cleaning
> are documented in **[data.md](data.md)**.

## 3. The Pipeline (Step by Step)

1. **Build the corpus** — [utils/build_multilingual_dataset.py](utils/build_multilingual_dataset.py)
   loads each MASSIVE language via the OOP `MassiveScenarioFetcher`, maps the scenario
   names to one shared integer label space, applies language-specific cleaning, and
   saves `data/processed_data/multilingual_corpus.csv`.
2. **Clean & normalize** — [utils/text_cleaning.py](utils/text_cleaning.py)
   (`LanguageSpecificSanitizer`): Unicode NFKC normalization that preserves `å ä ö`,
   diacritic protection for Swedish, suffix-preserving handling for Finnish, whitespace
   collapse. Light by design (no stemming/lowercasing) so the transformer keeps casing
   and context.
3. **Detect language** — [utils/language_detector.py](utils/language_detector.py)
   (`NgramLanguageDetector`) routes each document at inference time; training routes by
   the known `language` column.
4. **Tokenize** — [utils/tokenization.py](utils/tokenization.py): SpaCy word tokens for
   en/sv, WordPiece sub-words for agglutinative Finnish.
5. **Classify** — [models/text_classifier.py](models/text_classifier.py): a TF-IDF +
   Logistic Regression **baseline** (the floor) and a fine-tuned **DistilBERT**
   (multilingual) deep model, evaluated with accuracy, macro-F1, and AUC.
6. **Tag** — [models/tagger.py](models/tagger.py) (`MultilingualTagger`): SpaCy NER per
   language plus model-driven keyword extraction, normalized into one tag scheme.
7. **Serve in real time** — [models/pipeline.py](models/pipeline.py)
   (`RealTimePipeline`): detect → clean → classify (batched) → tag, returning one
   structured result per document, with a throughput benchmark.
8. **Optimize** — [models/optimization.py](models/optimization.py): magnitude pruning +
   float16 / int8 quantization for fast, small CPU/edge serving.
9. **Visualize** — [app/real_time_dashboard.py](app/real_time_dashboard.py): live
   analysis, performance metrics, and dataset distributions in Streamlit.

## 4. Results

### 4.1 Classification performance

From [reports/performance_metrics.json](reports/performance_metrics.json) (generated by
`python train.py`): **90.3 %** accuracy, **0.894** macro-F1, **305 docs/s**, with
per-language accuracy of **en 92.6 % · fi 89.3 % · sv 88.9 %** — all above the required
thresholds, and **+5.2 points** over the TF-IDF baseline (85.1 %).

### 4.2 Before vs after pruning + quantization

From [reports/optimization_metrics.json](reports/optimization_metrics.json) (generated
by `python optimize.py`):

| Variant | Size | vs original | Accuracy |
| --- | --- | --- | --- |
| Original (fp32) | 541 MB | — | 0.903 |
| **Pruned** (30 % magnitude sparsity) | 541 MB on disk · 40.6 M weights zeroed | highly compressible | **0.913** |
| **+ Float16** quantization | 200 MB | **−63 %** | 0.913 |
| **+ Int8** (TFLite, dynamic-range) | 137 MB | **−75 %** | converted ✓ |

Pruning 30 % of the weights cost **no accuracy** (it stayed ~0.90), and int8
quantization shrank the model **~4×**. The pruned weights reload into the *same*
architecture, so the serving pipeline uses them with no code change — just point
`SERVING_WEIGHTS_FILENAME` in [utils/config.py](utils/config.py) at the pruned file.

### 4.3 Why pruning/quantization matter for real-time systems

A real-time, high-volume document processor must keep **latency low**, **memory small**,
and ideally run on **cheap CPU/edge hardware** without a GPU. A 541 MB fp32 transformer
is memory-heavy and slow on CPU. Optimization addresses exactly this:

- **Pruning** zeros the least-important weights → a sparse model that **compresses far
  better** (faster to load/ship) and does **less work** per inference.
- **Quantization** stores weights in fewer bits (fp16 → int8) → **~4× smaller** and
  **faster integer math on CPUs**, which is where most production/edge serving happens.

Together they let the system hold the **≥ 100 docs/sec** bar and scale to high volume on
modest hardware, while keeping accuracy essentially unchanged — the core trade-off a
real-time AI system has to get right.

## 5. Usage

### 5.1 Prerequisites

- **Python 3.10–3.12** and a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

- **SpaCy language models** (needed for tagging and the dashboard):

```bash
python -m spacy download en_core_web_sm sv_core_news_sm fi_core_news_sm
```

- Quick sanity check: `python -c "import tensorflow, spacy, streamlit"`.

### 5.2 Run the dashboard

```bash
streamlit run app/real_time_dashboard.py
```

**The dashboard needs** (all produced by training):

- the trained checkpoint `models/checkpoints/text_classifier_best.h5` + `config.json`,
- the corpus `data/processed_data/multilingual_corpus.csv` (category names + Dataset tab),
- `reports/performance_metrics.json` (Performance tab),
- the SpaCy models from [§5.1](#51-prerequisites) (NER tagging).

It has three tabs: **Live Analysis** (paste a document → category, confidence, top-3
predictions, tags, entities, latency), **Performance** (accuracy, macro-F1, throughput,
per-language accuracy), and **Dataset** (category + language distributions). The served
model variant is shown in the Live tab and chosen via `SERVING_WEIGHTS_FILENAME` in
[utils/config.py](utils/config.py).

> 💡 MASSIVE is **short voice-assistant commands**, so the model is most meaningful on
> short, command-style inputs (e.g. *"set an alarm for seven am"*). Long articles fall
> outside the trained domain and trigger a low-confidence warning.

### 5.3 (Re)train the model

The corpus is committed, so you can train directly. To rebuild it first:

```bash
python -m utils.build_multilingual_dataset      # writes data/processed_data/multilingual_corpus.csv
```

Then train (GPU strongly recommended — install `tensorflow[and-cuda]` on a Linux NVIDIA
VM):

```bash
python train.py                 # 5 epochs on the full corpus
python train.py --epochs 5 --corpus data/processed_data/multilingual_corpus.csv
```

This writes `models/checkpoints/{text_classifier_best.h5, config.json,
training_history.csv}`, `reports/performance_metrics.json`, and
`reports/example_predictions.csv`.

### 5.4 Optimize the model (prune + quantize)

```bash
python optimize.py              # writes pruned/fp16/int8 artifacts + reports/optimization_metrics.json
```

Run the test suite with `pytest`.

## 6. Project Structure

```
document-categorization/
├── data/processed_data/        # multilingual_corpus.csv (built by the data builder)
├── models/
│   ├── text_classifier.py      # TF-IDF baseline + DistilBERT classifier
│   ├── tagger.py               # SpaCy NER + keyword tagging
│   ├── pipeline.py             # real-time detect → clean → classify → tag
│   ├── optimization.py         # pruning + quantization
│   └── checkpoints/            # trained + optimized model artifacts
├── utils/                      # config, data loader/fetcher/builder, cleaning, tokenization, language detection
├── app/real_time_dashboard.py  # Streamlit dashboard
├── notebooks/EDA_and_Training.ipynb
├── reports/                    # performance_metrics.json, optimization_metrics.json, example_predictions.csv
├── train.py                    # headless training pipeline
├── optimize.py                 # headless optimization pipeline
└── tests/                      # pytest suite
```

## 7. Further Documentation

- **[data.md](data.md)** — data source, EDA findings, and per-language cleaning.
- **[learning_objectives.md](learning_objectives.md)** — the NLP/ML concepts, and which
  are implemented vs theory-only.
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — coding standards, OOP/architecture, testing.
