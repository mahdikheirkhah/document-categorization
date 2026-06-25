# Data Documentation

End-to-end documentation of the dataset behind this project: **where the data
comes from**, **what we found when inspecting it**, and **how we cleaned it** for
each of the three supported languages — English, Swedish, and Finnish.

The pipeline follows **Route A**: take a recommended English dataset
(20 Newsgroups) and machine-translate a stratified sample into Swedish and Finnish
so all three languages share **one identical label space**. The data-acquisition
code lives in [utils/data_fetcher.py](utils/data_fetcher.py) and
[utils/build_multilingual_dataset.py](utils/build_multilingual_dataset.py); the EDA
and cleaning code live in the [EDA & Training notebook](notebooks/EDA_and_Training.ipynb.ipynb)
and [utils/text_cleaning.py](utils/text_cleaning.py).

---

## Table of Contents

- [1. Data Sources](#1-data-sources)
  - [1.1 English source: 20 Newsgroups](#11-english-source-20-newsgroups)
  - [1.2 Swedish (machine-translated)](#12-swedish-machine-translated)
  - [1.3 Finnish (machine-translated)](#13-finnish-machine-translated)
  - [1.4 Unified schema and storage](#14-unified-schema-and-storage)
- [2. Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
  - [2.1 What the EDA checks](#21-what-the-eda-checks)
    - [2.1.1 Missing values (NLP-aware)](#211-missing-values-nlp-aware)
    - [2.1.2 Duplicates](#212-duplicates)
    - [2.1.3 Document-length distribution and outliers](#213-document-length-distribution-and-outliers)
    - [2.1.4 Category distribution](#214-category-distribution)
    - [2.1.5 Language distribution](#215-language-distribution)
  - [2.2 English findings](#22-english-findings)
  - [2.3 Swedish findings](#23-swedish-findings)
  - [2.4 Finnish findings](#24-finnish-findings)
  - [2.5 Cross-language summary and challenges](#25-cross-language-summary-and-challenges)
- [3. Cleaning](#3-cleaning)
  - [3.1 Principles and strategy](#31-principles-and-strategy)
  - [3.2 Shared cleaning steps](#32-shared-cleaning-steps)
    - [3.2.1 HTML removal](#321-html-removal)
    - [3.2.2 Newsgroup noise removal](#322-newsgroup-noise-removal)
    - [3.2.3 URL and email removal](#323-url-and-email-removal)
    - [3.2.4 Unicode normalization](#324-unicode-normalization)
    - [3.2.5 Whitespace normalization](#325-whitespace-normalization)
  - [3.3 English](#33-english)
  - [3.4 Swedish](#34-swedish)
  - [3.5 Finnish](#35-finnish)
  - [3.6 Empty and micro-document filtering](#36-empty-and-micro-document-filtering)

---

## 1. Data Sources

This project supports three languages. English is sourced directly; Swedish and
Finnish are produced by machine translation so that every language shares the same
categories (see [§1.4](#14-unified-schema-and-storage)). All loading/building is
done with the OOP fetchers in [utils/data_fetcher.py](utils/data_fetcher.py).

### 1.1 English source: 20 Newsgroups

- **Where:** the [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/) corpus,
  pulled from the Hugging Face hub as `SetFit/20_newsgroups`.
- **How:** `EnglishNewsgroupsFetcher` (wraps `HuggingFaceCorpusLoader` in
  [utils/data_loader.py](utils/data_loader.py)).
- **Size / shape:** ~11,314 training documents across **20 categories** (e.g.
  `comp.graphics`, `rec.sport.hockey`, `sci.med`, `talk.politics.guns`). This is a
  *recommended* dataset for the project and comfortably exceeds the "≥10,000 docs,
  ≥5 categories" requirement.
- **Known characteristics:** raw posts carry email/news **headers, quoted reply
  lines, and signature blocks** — noise that must be removed before training
  (see [§2.2](#22-english-findings) and [§3.3](#33-english)).

### 1.2 Swedish (machine-translated)

- **Where:** translated from the English source with the free Opus-MT model
  [`Helsinki-NLP/opus-mt-en-sv`](https://huggingface.co/Helsinki-NLP/opus-mt-en-sv).
- **How:** `SwedishTranslationFetcher` in [utils/data_fetcher.py](utils/data_fetcher.py).
- **Why translation:** none of the recommended datasets (20 Newsgroups,
  Reuters-21578, MLDoc) contain Swedish, so we translate to keep an identical label
  space across languages (see [§1.4](#14-unified-schema-and-storage)).

### 1.3 Finnish (machine-translated)

- **Where:** translated from the English source with
  [`Helsinki-NLP/opus-mt-en-fi`](https://huggingface.co/Helsinki-NLP/opus-mt-en-fi).
- **How:** `FinnishTranslationFetcher` in [utils/data_fetcher.py](utils/data_fetcher.py).
- **Note:** translation runs in a **torch-only process** (`USE_TF=0`) because, on
  macOS, importing TensorFlow and PyTorch together aborts with an OpenMP
  `mutex lock failed`. Translation therefore happens in the offline builder, not in
  the training kernel.

### 1.4 Unified schema and storage

Every fetcher returns the same four-column schema so the languages concatenate
cleanly:

| Column | Description |
| --- | --- |
| `text` | The document content. |
| `label` | Integer category id (0–19). |
| `label_text` | Human-readable category name. |
| `language` | ISO 639-1 code: `en`, `sv`, or `fi`. |

Because the Swedish/Finnish rows are translations of the **same** English
documents, `label` / `label_text` are identical across languages — this is what
makes per-language accuracy comparable later.

- **Build command:** `python -m utils.build_multilingual_dataset --sample-size 300`
- **Output:** `data/processed_data/multilingual_corpus.csv`
- **Orchestration:** `MultilingualCorpusFetcher` combines the per-language fetchers.

---

## 2. Exploratory Data Analysis (EDA)

EDA is implemented by the `TextExploratoryAnalyzer` class in the
[EDA & Training notebook](notebooks/EDA_and_Training.ipynb.ipynb). It runs the same
analysis on each language so problems and differences are easy to compare. The
per-language numbers below come from the **aligned translated sample (96 documents
per language, same documents)**; the full English corpus is ~11,314 documents.

### 2.1 What the EDA checks

#### 2.1.1 Missing values (NLP-aware)

In NLP a row is "missing" not only when it is `NaN`. We classify every document
into four mutually-exclusive buckets: **NaN/Null**, **Empty/Whitespace** ("ghost"
docs), **Micro** (< 3 words, too little context for a transformer), and **Valid**.
Addressed by cleaning step [§3.6](#36-empty-and-micro-document-filtering).

#### 2.1.2 Duplicates

Exact-duplicate documents inflate metrics and leak across the train/test split.

#### 2.1.3 Document-length distribution and outliers

Word-count distribution with `describe()` plus **IQR-based** outlier detection
(histogram + boxplot). Extreme length skew distorts the feature space and motivates
truncation at the tokenizer level.

#### 2.1.4 Category distribution

Per-category counts, the most/least common label, and the **imbalance ratio**
(max/min). Drives the use of macro-F1 and class weights during training.

#### 2.1.5 Language distribution

Document counts per language — confirms the multi-language coverage required by the
subject and feeds the dashboard's language breakdown.

### 2.2 English findings

- **Missing values:** 0 NaN, **3 empty/whitespace ghosts**, 1 micro, 92 valid.
- **Duplicates:** 2 (2.08%).
- **Length (words):** median **94.5**, mean **166** — heavily right-skewed,
  **max 3065 words**; 4 IQR outliers (~4.35%).
- **Categories:** 20, mild imbalance — most common `comp.graphics` (5),
  least common `talk.religion.misc` (3), **imbalance ratio 1.67**.
- **Main challenge:** pervasive **header / quoted-reply / signature noise** that
  leaks labels and dilutes signal → see cleaning [§3.3](#33-english).

### 2.3 Swedish findings

- **Missing values:** 0 NaN, **0 empty** — note this is *worse than it looks*: the
  empty English docs were not preserved but **hallucinated** into content by the
  translator (e.g. an empty input became `"- Jag vet inte."`). Handled by filtering
  before translation, see [§3.6](#36-empty-and-micro-document-filtering).
- **Length (words):** median **87.0**, mean **83**, **max 190** — close to English.
- **Challenge:** the maximum is **capped by the Opus-MT 512-token input limit**, so
  long English posts are truncated in translation (0 length-outliers is an
  artifact, not linguistics).
- **Categories:** identical to English (imbalance 1.67), by construction.

### 2.4 Finnish findings

- **Missing values:** 0 NaN, 0 empty (same hallucination caveat as Swedish;
  empties became `"- Ei, ei, ei..."` degeneration).
- **Length (words):** median **56.0**, mean **64**, max 331 — **~40% fewer words**
  than English for the *same* documents. This is the **agglutinative** nature of
  Finnish: case, number, and possession fold into single words.
- **Challenge:** same 512-token translation cap; and downstream tokenization must be
  **sub-word** (WordPiece) rather than word-level — handled in
  [utils/tokenization.py](utils/tokenization.py), not in cleaning.
- **Categories:** identical to English (imbalance 1.67).

### 2.5 Cross-language summary and challenges

| Metric (96-doc aligned sample) | English | Swedish | Finnish |
| --- | --- | --- | --- |
| Median words/doc | 94.5 | 87.0 | 56.0 |
| Mean words/doc | 166 | 83 | 64 |
| Max words/doc | 3065 | 190 | 331 |
| NaN / empty / micro | 0 / 3 / 1 | 0 / 0 / 1 | 0 / 0 / 1 |
| Duplicates | 2 | 2 | 2 |
| Categories / imbalance | 20 / 1.67 | 20 / 1.67 | 20 / 1.67 |

Challenges that drive the cleaning design in [§3](#3-cleaning):

1. **Newsgroup noise** (English) — headers, quotes, signatures → [§3.2.2](#322-newsgroup-noise-removal).
2. **Empty/micro docs + translation hallucination** → [§3.6](#36-empty-and-micro-document-filtering).
3. **Encoding / diacritics** (`å ä ö`) must survive → [§3.2.4](#324-unicode-normalization).
4. **Length skew / outliers** — not "cleaned"; handled by tokenizer truncation at
   training time (`max_length`).
5. **Mild class imbalance (1.67)** — not a cleaning issue; handled at training via
   `class_weight` and macro-F1 evaluation.

---

## 3. Cleaning

Cleaning is implemented in [utils/text_cleaning.py](utils/text_cleaning.py) and
applied per document in the notebook, **routing by the known `language` column**
(no runtime language detection during training — that belongs to inference).

### 3.1 Principles and strategy

- **Light cleaning for transformers.** DistilBERT relies on casing and context and
  uses sub-word tokenization, so we deliberately **do not** lowercase, remove
  stopwords, or stem. Aggressive normalization hurts the model. (The TF-IDF baseline
  separately gets lowercasing/stopwords from `TfidfVectorizer`.)
- **One unified pipeline.** `LanguageSpecificSanitizer` applies the same shared
  steps ([§3.2](#32-shared-cleaning-steps)) to every language, then a light
  language-specific tweak ([§3.3](#33-english)–[§3.5](#35-finnish)).
- **Each step maps to a finding** in [§2.5](#25-cross-language-summary-and-challenges).

### 3.2 Shared cleaning steps

Applied to **all** languages, in this order: HTML → newsgroup noise → URL/email →
unicode → (language tweak) → whitespace.

#### 3.2.1 HTML removal

`HtmlCleaner` strips HTML tags. Cheap insurance against markup artifacts in any
language.

#### 3.2.2 Newsgroup noise removal

`NewsgroupNoiseCleaner` removes email/news **headers** (`From:`, `Subject:`, …),
**quoted reply lines** (`>`, `>>`, `RB>`, `: >`), **attribution lines**
(`In article … writes:`), and **signature blocks** (after `-- `). This is the
single highest-impact step for the English corpus
(finding [§2.2](#22-english-findings)) — equivalent to scikit-learn's
`remove=('headers', 'footers', 'quotes')`. On a sample post it cut a 480-char
document to 95 chars of real content.

#### 3.2.3 URL and email removal

`ArtifactCleaner` removes URLs and email addresses, which carry no categorical
signal.

#### 3.2.4 Unicode normalization

`UnicodeNormalizer` applies **NFKC** normalization and logs non-ASCII counts for
observability. Critical for preserving Nordic characters `å ä ö`
(challenge [§2.5](#25-cross-language-summary-and-challenges), item 3).

#### 3.2.5 Whitespace normalization

`DuplicationCleaner` collapses repeated whitespace and trims edges — always runs
last so earlier removals don't leave ragged gaps.

### 3.3 English

- **Pipeline:** shared steps [§3.2](#32-shared-cleaning-steps) only (no extra tweak).
- **Why:** the dominant English problem is structural newsgroup noise
  ([§2.2](#22-english-findings)), fully handled by
  [§3.2.2](#322-newsgroup-noise-removal). No diacritic handling needed.

### 3.4 Swedish

- **Pipeline:** shared steps + `SwedishCompoundCleaner` (NFC) to keep combined
  diacritics `å ä ö` composed and protect compound-word integrity.
- **Why:** Swedish text is otherwise close to English; the focus is purely on not
  corrupting diacritics (finding [§2.3](#23-swedish-findings)).

### 3.5 Finnish

- **Pipeline:** shared steps + `FinnishMorphologyCleaner` (conservative — strip only).
- **Why:** Finnish is **agglutinative** (finding [§2.4](#24-finnish-findings)); we
  must **not** stem or strip suffixes, as meaning lives in stacked morphemes.
  Word-segmentation is deferred to **sub-word (WordPiece) tokenization** in
  [utils/tokenization.py](utils/tokenization.py), not done here.

### 3.6 Empty and micro-document filtering

Empty/whitespace ("ghost") and micro (< 3 words) documents
(finding [§2.1.1](#211-missing-values-nlp-aware)) are removed at **two points**:

1. **Before translation**, in
   [utils/build_multilingual_dataset.py](utils/build_multilingual_dataset.py) — an
   empty input otherwise makes the translator **hallucinate**
   (e.g. `"- Ei, ei, ei..."`), so we drop those English rows up front.
2. **After cleaning**, during training preprocessing — a document that was *all*
   quotes/headers can become empty once [§3.2.2](#322-newsgroup-noise-removal) runs,
   so any doc with fewer than 3 words after cleaning is dropped before the split.
