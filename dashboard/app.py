from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MT Quality Evaluation",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "experimental"
    / "samanantar_en_ml_quality_scored_1k.parquet"
)


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🌐 Machine Translation Quality Evaluation")

st.markdown(
    """
    ### English → Malayalam

    This dashboard presents the quality evaluation of machine
    translated Malayalam sentences using three complementary
    evaluation dimensions:

    - **Lexical Evaluation**
    - **Structural Evaluation**
    - **Semantic Evaluation**

    These dimensions are combined into a final MT quality score.
    """
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not DATA_FILE.exists():
        return None

    df = pd.read_parquet(DATA_FILE)

    return df


df = load_data()


if df is None:

    st.error(
        "Final dataset was not found.\n\n"
        f"Expected location:\n{DATA_FILE}"
    )

    st.stop()


# ============================================================
# QUALITY BAND
# ============================================================

if "final_quality_score" in df.columns:

    def quality_band(score):

        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Fair"
        else:
            return "Poor"

    df["quality_band"] = (
        df["final_quality_score"]
        .apply(quality_band)
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Dashboard Filters")

st.sidebar.write(
    f"Dataset contains **{len(df):,} sentences**."
)


# Minimum quality score filter

if "final_quality_score" in df.columns:

    min_score = st.sidebar.slider(
        "Minimum Final Quality Score",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0
    )

else:

    min_score = 0


# Quality band filter

if "quality_band" in df.columns:

    band_options = [
        "All",
        "Excellent",
        "Good",
        "Fair",
        "Poor"
    ]

    selected_band = st.sidebar.selectbox(
        "Quality Band",
        band_options
    )

else:

    selected_band = "All"


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()


if "final_quality_score" in filtered_df.columns:

    filtered_df = filtered_df[
        filtered_df["final_quality_score"] >= min_score
    ]


if (
    selected_band != "All"
    and "quality_band" in filtered_df.columns
):

    filtered_df = filtered_df[
        filtered_df["quality_band"] == selected_band
    ]


# ============================================================
# OVERVIEW METRICS
# ============================================================

st.subheader("📊 Overall Evaluation")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Sentences",
        f"{len(filtered_df):,}"
    )


with col2:

    if "final_quality_score" in filtered_df.columns:

        st.metric(
            "Average Quality",
            f"{filtered_df['final_quality_score'].mean():.2f}"
        )


with col3:

    if "semantic_similarity" in filtered_df.columns:

        st.metric(
            "Average Semantic Similarity",
            f"{filtered_df['semantic_similarity'].mean():.3f}"
        )


with col4:

    if "unigram_f1" in filtered_df.columns:

        st.metric(
            "Average Lexical F1",
            f"{filtered_df['unigram_f1'].mean():.3f}"
        )


# ============================================================
# QUALITY DISTRIBUTION
# ============================================================

st.subheader("📈 Quality Distribution")

if "final_quality_score" in filtered_df.columns:

    fig = px.histogram(
        filtered_df,
        x="final_quality_score",
        nbins=30,
        title="Final MT Quality Score Distribution"
    )

    fig.update_layout(
        xaxis_title="Final Quality Score",
        yaxis_title="Number of Sentences"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# QUALITY BAND DISTRIBUTION
# ============================================================

if "quality_band" in filtered_df.columns:

    st.subheader("🏷️ Quality Bands")

    band_counts = (
        filtered_df["quality_band"]
        .value_counts()
        .reindex(
            ["Excellent", "Good", "Fair", "Poor"],
            fill_value=0
        )
        .reset_index()
    )

    band_counts.columns = [
        "Quality Band",
        "Number of Sentences"
    ]

    fig = px.bar(
        band_counts,
        x="Quality Band",
        y="Number of Sentences",
        title="MT Quality Band Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# LEXICAL VS SEMANTIC
# ============================================================

if (
    "unigram_f1" in filtered_df.columns
    and "semantic_similarity" in filtered_df.columns
):

    st.subheader("🔤 Lexical vs Semantic Evaluation")

    fig = px.scatter(
        filtered_df,
        x="unigram_f1",
        y="semantic_similarity",
        hover_data=[
            column
            for column in [
                "idx",
                "source",
                "reference",
                "translation",
                "final_quality_score"
            ]
            if column in filtered_df.columns
        ],
        title="Lexical F1 vs Semantic Similarity"
    )

    fig.update_layout(
        xaxis_title="Unigram F1",
        yaxis_title="Semantic Similarity"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# SCORE COMPONENTS
# ============================================================

st.subheader("⚖️ Evaluation Dimensions")

score_columns = [
    "lexical_quality_score",
    "structural_quality_score",
    "semantic_similarity"
]

available_scores = [
    column
    for column in score_columns
    if column in filtered_df.columns
]

if available_scores:

    score_values = {
        "Lexical": (
            filtered_df["lexical_quality_score"].mean()
            if "lexical_quality_score" in filtered_df.columns
            else None
        ),
        "Structural": (
            filtered_df["structural_quality_score"].mean()
            if "structural_quality_score" in filtered_df.columns
            else None
        ),
        "Semantic": (
            filtered_df["semantic_similarity"].mean()
            if "semantic_similarity" in filtered_df.columns
            else None
        )
    }

    score_values = {
        key: value
        for key, value in score_values.items()
        if value is not None
    }

    score_df = pd.DataFrame(
        {
            "Dimension": list(score_values.keys()),
            "Average Score": list(score_values.values())
        }
    )

    fig = px.bar(
        score_df,
        x="Dimension",
        y="Average Score",
        title="Average Score by Evaluation Dimension"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# SENTENCE EXPLORER
# ============================================================

st.subheader("🔎 Sentence-Level Evaluation")

if "idx" in filtered_df.columns:

    selected_idx = st.selectbox(
        "Select a sentence",
        filtered_df["idx"].tolist()
    )

    selected_row = filtered_df[
        filtered_df["idx"] == selected_idx
    ].iloc[0]

else:

    selected_row = filtered_df.iloc[0]


# ============================================================
# TEXT DISPLAY
# ============================================================

text_col1, text_col2, text_col3 = st.columns(3)


with text_col1:

    st.markdown("### 🇬🇧 Source")

    st.info(
        str(
            selected_row.get(
                "source",
                "Not available"
            )
        )
    )


with text_col2:

    st.markdown("### 🇮🇳 Reference Malayalam")

    st.info(
        str(
            selected_row.get(
                "reference",
                "Not available"
            )
        )
    )


with text_col3:

    st.markdown("### 🤖 Machine Translation")

    st.info(
        str(
            selected_row.get(
                "translation",
                "Not available"
            )
        )
    )


# ============================================================
# SELECTED SENTENCE SCORES
# ============================================================

st.subheader("📋 Sentence Evaluation Scores")

metric_columns = [
    "unigram_precision",
    "unigram_recall",
    "unigram_f1",
    "sentence_bleu",
    "edit_similarity",
    "length_similarity",
    "punctuation_similarity",
    "semantic_similarity",
    "lexical_quality_score",
    "structural_quality_score",
    "final_quality_score"
]

available_metrics = [
    column
    for column in metric_columns
    if column in selected_row.index
]


metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

for i, column in enumerate(available_metrics):

    value = selected_row[column]

    if pd.isna(value):
        display_value = "N/A"
    else:
        display_value = f"{float(value):.4f}"

    target_column = [
        metric_col1,
        metric_col2,
        metric_col3,
        metric_col4
    ][i % 4]

    readable_name = (
        column
        .replace("_", " ")
        .title()
    )

    target_column.metric(
        readable_name,
        display_value
    )


# ============================================================
# TOP / LOW QUALITY EXAMPLES
# ============================================================

st.subheader("🏆 Highest Quality Examples")

if "final_quality_score" in filtered_df.columns:

    top_examples = (
        filtered_df
        .sort_values(
            "final_quality_score",
            ascending=False
        )
        .head(10)
    )

    display_columns = [
        column
        for column in [
            "idx",
            "source",
            "reference",
            "translation",
            "semantic_similarity",
            "final_quality_score"
        ]
        if column in top_examples.columns
    ]

    st.dataframe(
        top_examples[display_columns],
        use_container_width=True,
        hide_index=True
    )


st.subheader("⚠️ Lowest Quality Examples")

if "final_quality_score" in filtered_df.columns:

    poor_examples = (
        filtered_df
        .sort_values(
            "final_quality_score",
            ascending=True
        )
        .head(10)
    )

    display_columns = [
        column
        for column in [
            "idx",
            "source",
            "reference",
            "translation",
            "semantic_similarity",
            "final_quality_score"
        ]
        if column in poor_examples.columns
    ]

    st.dataframe(
        poor_examples[display_columns],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# COMPLETE DATASET
# ============================================================

with st.expander("📄 View Filtered Dataset"):

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Machine Translation Quality Evaluation | "
    "English → Malayalam | NLP Project"
)