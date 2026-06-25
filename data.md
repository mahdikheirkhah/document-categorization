# Data Documentation

End-to-end documentation of the dataset behind this project: **where the data
comes from**, **what we found when inspecting it**, and **how we cleaned it** for
each of the three supported languages — English, Swedish, and Finnish.

The project uses the **MASSIVE** dataset (Amazon): a *natively multilingual,
parallel* corpus where the same utterances are provided in every language, each
labelled with one of **18 "scenario" domains** (`alarm`, `calendar`, `weather`, …).
Because the languages are parallel, Swedish and Finnish are **real native text**
(not machine translations) and all three languages share **one identical label
space** — with no translation step. The data-acquisition code lives in
[utils/data_fetcher.py](utils/data_fetcher.py) and
[utils/build_multilingual_dataset.py](utils/build_multilingual_dataset.py); the EDA
and cleaning code live in the [EDA & Training notebook](notebooks/EDA_and_Training.ipynb)
and [utils/text_cleaning.py](utils/text_cleaning.py).

---

## Table of Contents

- [1. Data Sources](#1-data-sources)
  - [1.1 Source: MASSIVE (scenario)](#11-source-massive-scenario)
  - [1.2 Why MASSIVE for English + Swedish + Finnish](#12-why-massive-for-english--swedish--finnish)
  - [1.3 Unified schema and storage](#13-unified-schema-and-storage)
- [2. Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
  - [2.1 What the EDA checks](#21-what-the-eda-checks)
  - [2.2 Findings (per language)](#22-findings-per-language)
  - [2.3 Cross-language summary and challenges](#23-cross-language-summary-and-challenges)
- [3. Cleaning](#3-cleaning)
  - [3.1 Principles and strategy](#31-principles-and-strategy)
  - [3.2 Shared cleaning steps](#32-shared-cleaning-steps)
  - [3.3 Per-language tweaks](#33-per-language-tweaks)
  - [3.4 Empty-document filtering](#34-empty-document-filtering)

---

## 1. Data Sources

This project supports three languages in **one** dataset, so no source-specific
glue or translation is needed. All loading/building is done with the OOP fetchers
in [utils/data_fetcher.py](utils/data_fetcher.py).

### 1.1 Source: MASSIVE (scenario)

- **Where:** the [MASSIVE](https://huggingface.co/datasets/mteb/amazon_massive_scenario)
  dataset (Amazon), pulled from the Hugging Face hub as
  `mteb/amazon_massive_scenario`. We use the **MTEB parquet mirror** because the
  canonical `AmazonScience/massive` repo ships a dataset *script*, which the
  `datasets>=4` library no longer executes.
- **How:** `MassiveScenarioFetcher` (wraps `HuggingFaceCorpusLoader` in
  [utils/data_loader.py](utils/data_loader.py)), one instance per language
  (`en`, `sv`, `fi`), combined by `MultilingualCorpusFetcher`.
- **Task / labels:** we classify the **18 coarse `scenario` domains** (`alarm`,
  `audio`, `calendar`, `cooking`, `datetime`, `email`, `general`, `iot`, `lists`,
  `music`, `news`, `play`, `qa`, `recommendation`, `social`, `takeaway`,
  `transport`, `weather`) — not the 60 fine intents. Fewer, better-separated
  classes make the ≥85% accuracy / ≥0.80 macro-F1 targets realistic.
- **Size / shape:** **11,514 documents per language** (the `train` split),
  **34,542 documents total** across `en` + `sv` + `fi`. This comfortably exceeds
  the "≥10,000 docs, ≥5 categories, ≥2 languages" requirement.
- **Known characteristics:** documents are **short voice-assistant utterances**
  (e.g. *"wake me up at nine am on friday"*), already lowercased and largely free
  of HTML/markup. Median length is ~6 words in English/Swedish and ~4 in Finnish
  (see [§2](#2-exploratory-data-analysis-eda)).

### 1.2 Why MASSIVE for English + Swedish + Finnish

None of the *recommended* datasets (20 Newsgroups, Reuters-21578, MLDoc) contain
Swedish **or** Finnish. MASSIVE does, natively and in parallel, which means:

- **One identical label space** across all three languages, by construction — no
  translation, no label drift.
- **Honest per-language accuracy:** because the categories are identical, the
  ≥80%-per-language requirement is directly comparable across `en`, `sv`, `fi`.
- **Real native text:** Swedish and Finnish are written by native speakers, so
  diacritics (`å ä ö`) and Finnish agglutination are genuine, not translation
  artifacts.

### 1.3 Unified schema and storage

Every fetcher returns the same four-column schema so the languages concatenate
cleanly:

| Column | Description |
| --- | --- |
| `text` | The utterance content. |
| `label` | Integer category id (0–17). |
| `label_text` | Human-readable scenario name (e.g. `calendar`). |
| `language` | ISO 639-1 code: `en`, `sv`, or `fi`. |

The scenario→id map is built deterministically (scenario names sorted
alphabetically) from English and **injected** into the Swedish and Finnish
fetchers, guaranteeing `label` / `label_text` are identical across languages.

- **Build command:** `python -m utils.build_multilingual_dataset` (add
  `--sample-size 1200` for a quick, stratified build)
- **Output:** `data/processed_data/multilingual_corpus.csv` (full corpus) and
  `data/processed_data/multilingual_sample.csv` (100 rows/language demo slice)
- **Orchestration:** `MultilingualCorpusFetcher` combines the per-language fetchers.

---

## 2. Exploratory Data Analysis (EDA)

EDA is implemented by the `TextExploratoryAnalyzer` class in the
[EDA & Training notebook](notebooks/EDA_and_Training.ipynb). It runs the same
analysis on each language so problems and differences are easy to compare. The
numbers below are computed over the **full 11,514-document `train` split per
language**.

### 2.1 What the EDA checks

- **Missing values (NLP-aware).** A row is "missing" not only when it is `NaN`, but
  also when it is **empty/whitespace** ("ghost") or **micro** (too few words for
  context). MASSIVE utterances are short by design, so the micro threshold is set
  to `min_words=1` — only truly empty rows count.
- **Duplicates.** Exact-duplicate utterances inflate metrics and leak across the
  train/test split. They are common here because many commands are short and
  naturally repeat (e.g. *"stop"*, *"pysäytä"*).
- **Document-length distribution and outliers.** Word-count `describe()` plus
  IQR-based outlier detection (histogram + boxplot).
- **Category distribution.** Per-category counts, most/least common label, and the
  **imbalance ratio** (max/min) — drives macro-F1 and class weights at training.
- **Language distribution.** Document counts per language — confirms the
  multi-language coverage and feeds the dashboard's language breakdown.

### 2.2 Findings (per language)

- **English** — 0 empty, **46 duplicates**, median **6** words (mean 6.9, max 35).
- **Swedish** — 0 empty, **407 duplicates**, median **6** words (mean 6.3, max 29);
  diacritics `å ä ö` present and preserved.
- **Finnish** — 0 empty, **414 duplicates**, median **4** words (mean 4.8, max 26) —
  **~30% fewer words** than English for the *same* utterances, reflecting Finnish
  **agglutination** (case, number and possession fold into single words).

All three languages share the **same 18 categories** with the **same imbalance
ratio of 8.0** (largest `calendar` = 1,688 docs; smallest `cooking` = 211 docs),
because the corpus is parallel.

### 2.3 Cross-language summary and challenges

| Metric (full train split) | English | Swedish | Finnish |
| --- | --- | --- | --- |
| Documents | 11,514 | 11,514 | 11,514 |
| Median words/doc | 6 | 6 | 4 |
| Mean words/doc | 6.9 | 6.3 | 4.8 |
| Max words/doc | 35 | 29 | 26 |
| Empty / duplicates | 0 / 46 | 0 / 407 | 0 / 414 |
| Categories / imbalance | 18 / 8.0 | 18 / 8.0 | 18 / 8.0 |

Challenges that drive the cleaning design in [§3](#3-cleaning):

1. **Diacritics** (`å ä ö`) must survive intact → Unicode normalization
   ([§3.2](#32-shared-cleaning-steps)).
2. **Short utterances** — no aggressive filtering; a 2-word command like *"play
   music"* is valid → `min_words=1` ([§3.4](#34-empty-document-filtering)).
3. **Finnish agglutination** — never stem/strip suffixes; word-segmentation is
   deferred to **sub-word (WordPiece) tokenization** in
   [utils/tokenization.py](utils/tokenization.py).
4. **Class imbalance (8.0)** — not a cleaning issue; handled at training via
   `class_weight="balanced"` and macro-F1 evaluation.

---

## 3. Cleaning

Cleaning is implemented in [utils/text_cleaning.py](utils/text_cleaning.py) and
applied per language in [utils/build_multilingual_dataset.py](utils/build_multilingual_dataset.py),
**routing by the known `language` column** (no runtime language detection during
the offline build — that belongs to inference).

### 3.1 Principles and strategy

- **Light cleaning for transformers.** DistilBERT relies on casing and context and
  uses sub-word tokenization, so we deliberately **do not** lowercase, remove
  stopwords, or stem. (The TF-IDF baseline separately gets lowercasing/stopwords
  from `TfidfVectorizer`.) MASSIVE text is already clean, so cleaning is mostly
  Unicode/whitespace hygiene.
- **One unified pipeline.** `LanguageSpecificSanitizer` applies the same shared
  steps to every language, then a light language-specific tweak.

### 3.2 Shared cleaning steps

Applied to **all** languages, in order: HTML strip → structural-noise strip →
URL/email strip → Unicode normalization → (language tweak) → whitespace collapse.
The HTML / structural-noise / URL / email cleaners are inexpensive insurance — they
rarely match the short, clean MASSIVE utterances but guard against stray artifacts.
The load-bearing step here is **Unicode normalization** (`UnicodeNormalizer`, NFKC),
which preserves Nordic characters `å ä ö` and logs non-ASCII counts for
observability; `DuplicationCleaner` then collapses repeated whitespace and trims
edges last.

### 3.3 Per-language tweaks

- **English** — shared steps only (no diacritics to protect).
- **Swedish** — `SwedishCompoundCleaner` (NFC) keeps combined diacritics `å ä ö`
  composed and protects compound-word integrity.
- **Finnish** — `FinnishMorphologyCleaner` (conservative — strip only): Finnish is
  **agglutinative**, so we must not stem or strip suffixes; meaning lives in stacked
  morphemes. Word-segmentation is deferred to sub-word tokenization in
  [utils/tokenization.py](utils/tokenization.py).

### 3.4 Empty-document filtering

After cleaning, any document with fewer than `MIN_WORDS` (= **1**) words — i.e.
empty/whitespace-only "ghost" rows — is dropped. The threshold is intentionally low
because MASSIVE utterances are short commands; a 1–2 word utterance is legitimate
content, unlike in a long-document corpus.
