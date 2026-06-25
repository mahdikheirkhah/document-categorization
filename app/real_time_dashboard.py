"""
app/real_time_dashboard.py

Streamlit dashboard for the multi-language document categorization & tagging system.

Run with:
    streamlit run app/real_time_dashboard.py

Three tabs:
    * Live Analysis  — paste a document -> category, confidence, top-3 predictions,
      tags, named entities, and latency (real-time categorization + tagging).
    * Performance    — accuracy, macro-F1, throughput, per-language accuracy, and
      the lift over the baseline (from reports/performance_metrics.json).
    * Dataset        — category, language, and (session) tag-count distributions.

Heavy resources (the trained model, corpus, metrics) are cached so the app stays
responsive; the model is loaded lazily on first analysis.
"""
import os

# Keep transformers TensorFlow-only (the classifier is TF/Keras).
os.environ.setdefault("USE_TORCH", "0")

import json
import time
from collections import Counter

import pandas as pd
import streamlit as st
from loguru import logger

from models.pipeline import RealTimePipeline

METRICS_PATH = "reports/performance_metrics.json"
CORPUS_PATH = "data/processed_data/multilingual_corpus.csv"


@st.cache_resource(show_spinner="Loading the trained model (one-time)...")
def get_pipeline() -> RealTimePipeline:
    """Loads and caches the RealTimePipeline (model loaded once across reruns)."""
    return RealTimePipeline()


@st.cache_data
def get_metrics() -> dict:
    """Loads the saved performance metrics, or None if not yet generated."""
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Metrics not available: {e}")
        return None


@st.cache_data
def get_corpus() -> pd.DataFrame:
    """Loads category/language columns of the corpus for charts, or None if absent."""
    try:
        return pd.read_csv(CORPUS_PATH, usecols=["label_text", "language"])
    except Exception as e:
        logger.warning(f"Corpus not available: {e}")
        return None


def render_live_analysis() -> None:
    """Renders the interactive categorization + tagging panel."""
    st.subheader("Live Categorization & Tagging")
    text = st.text_area(
        "Paste a document",
        height=200,
        placeholder="Type or paste text in English, Swedish, or Finnish...",
    )

    if st.button("Analyze", type="primary"):
        if not text.strip():
            st.warning("Please paste some text first.")
        else:
            try:
                pipeline = get_pipeline()
                start = time.time()
                result = pipeline.process(text)
                latency_ms = (time.time() - start) * 1000.0

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Category", result["category"])
                col2.metric("Confidence", f"{result['confidence']:.1%}")
                col3.metric("Language", result["language"])
                col4.metric("Latency", f"{latency_ms:.0f} ms")

                st.markdown("**Why — top predictions**")
                st.bar_chart(
                    pd.Series(
                        {t["category"]: t["probability"] for t in result["top_categories"]}
                    )
                )

                st.markdown("**Tags**")
                st.write(" ".join(f"`{tag}`" for tag in result["tags"]) or "_no tags_")

                if result["entities"]:
                    st.markdown("**Named entities (NER)**")
                    st.dataframe(
                        pd.DataFrame(result["entities"]),
                        use_container_width=True,
                        hide_index=True,
                    )

                # Accumulate for the session-level monitoring chart.
                st.session_state.setdefault("tag_history", []).extend(result["tags"])
            except Exception as e:
                logger.error(f"Live analysis failed: {e}")
                st.error(
                    f"Analysis failed: {e}\n\n"
                    "If the model is missing, train it first with `python train.py`."
                )

    if st.session_state.get("tag_history"):
        st.markdown("**Tag counts (this session)**")
        counts = pd.Series(Counter(st.session_state["tag_history"])).sort_values(ascending=False)
        st.bar_chart(counts)


def render_performance() -> None:
    """Renders the saved performance metrics and per-language accuracy."""
    st.subheader("Performance Metrics")
    metrics = get_metrics()
    if not metrics:
        st.info("No metrics yet — run `python train.py` to generate reports/performance_metrics.json.")
        return

    accuracy = metrics["classification_accuracy"]
    baseline = metrics.get("baseline_accuracy", 0.0)
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy:.1%}")
    col2.metric("Macro-F1", f"{metrics['f1_score_macro']:.3f}")
    col3.metric("Speed", f"{metrics['processing_speed_docs_per_sec']:.0f} docs/s")

    col4, col5 = st.columns(2)
    col4.metric("Baseline accuracy", f"{baseline:.1%}")
    col5.metric("Lift over baseline", f"+{(accuracy - baseline) * 100:.0f} pts")

    st.markdown("**Per-language accuracy** (target ≥ 80%)")
    st.bar_chart(pd.Series(metrics["per_language_accuracy"]))


def render_dataset() -> None:
    """Renders dataset category and language distributions."""
    st.subheader("Dataset Overview")
    corpus = get_corpus()
    if corpus is None:
        st.info("Corpus not found at data/processed_data/multilingual_corpus.csv.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Category distribution**")
        st.bar_chart(corpus["label_text"].value_counts())
    with col2:
        st.markdown("**Language distribution**")
        st.bar_chart(corpus["language"].value_counts())


def main() -> None:
    """Lays out the dashboard."""
    st.set_page_config(page_title="Document Categorization & Tagging", layout="wide")
    st.title("Document Categorization & Tagging — Real-Time Dashboard")
    st.caption(
        "Multi-language (English · Swedish · Finnish) — DistilBERT classifier + SpaCy NER tagging"
    )
    tab_live, tab_perf, tab_data = st.tabs(["Live Analysis", "Performance", "Dataset"])
    with tab_live:
        render_live_analysis()
    with tab_perf:
        render_performance()
    with tab_data:
        render_dataset()


main()