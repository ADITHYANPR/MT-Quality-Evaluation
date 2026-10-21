from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "experimental"
    / "samanantar_en_ml_quality_scored_1k.parquet"
)

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Final dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    print("=" * 70)
    print("FINAL DATASET LOADED")
    print("=" * 70)
    print("Shape:", df.shape)
    print("Columns:", len(df.columns))
    print()

    return df


# ============================================================
# HELPER
# ============================================================

def save_histogram(df, column, title, filename, xlabel):
    if column not in df.columns:
        print(f"Skipping {column}: column not found.")
        return

    plt.figure(figsize=(9, 6))

    sns.histplot(
        df[column].dropna(),
        bins=30,
        kde=True
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of sentences")
    plt.tight_layout()

    output = FIGURES_DIR / filename
    plt.savefig(output, dpi=300)
    plt.close()

    print("Saved:", output)


# ============================================================
# SCORE DISTRIBUTIONS
# ============================================================

def create_score_distributions(df):

    save_histogram(
        df,
        "unigram_f1",
        "Distribution of Unigram F1",
        "lexical_score_distribution.png",
        "Unigram F1"
    )

    save_histogram(
        df,
        "length_similarity",
        "Distribution of Length Similarity",
        "structural_score_distribution.png",
        "Length Similarity"
    )

    save_histogram(
        df,
        "semantic_similarity",
        "Distribution of Semantic Similarity",
        "semantic_score_distribution.png",
        "Semantic Similarity"
    )

    save_histogram(
        df,
        "final_quality_score",
        "Distribution of Final MT Quality Score",
        "final_quality_distribution.png",
        "Final Quality Score"
    )


# ============================================================
# CORRELATION HEATMAP
# ============================================================

def create_correlation_heatmap(df):

    important_columns = [
        "unigram_precision",
        "unigram_recall",
        "unigram_f1",
        "sentence_bleu",
        "edit_similarity",
        "length_similarity",
        "punctuation_similarity",
        "translation_repetition_ratio",
        "semantic_similarity",
    ]

    available_columns = [
        column
        for column in important_columns
        if column in df.columns
    ]

    if len(available_columns) < 2:
        print("Not enough columns for correlation analysis.")
        return

    correlation = df[available_columns].corr(method="pearson")

    correlation_file = TABLES_DIR / "feature_correlations.csv"
    correlation.to_csv(correlation_file)

    print("Saved:", correlation_file)

    plt.figure(figsize=(13, 10))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True
    )

    plt.title("Correlation Between MT Evaluation Features")
    plt.tight_layout()

    output = FIGURES_DIR / "correlation_heatmap.png"
    plt.savefig(output, dpi=300)
    plt.close()

    print("Saved:", output)

# ============================================================
# LEXICAL VS SEMANTIC
# ============================================================

def create_lexical_vs_semantic(df):

    required = [
        "unigram_f1",
        "semantic_similarity"
    ]

    if not all(column in df.columns for column in required):
        print("Skipping lexical vs semantic plot.")
        return

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=df,
        x="unigram_f1",
        y="semantic_similarity",
        alpha=0.6
    )

    plt.title("Lexical F1 vs Semantic Similarity")
    plt.xlabel("Unigram F1")
    plt.ylabel("Semantic Similarity")
    plt.tight_layout()

    output = FIGURES_DIR / "lexical_vs_semantic.png"
    plt.savefig(output, dpi=300)
    plt.close()

    print("Saved:", output)


# ============================================================
# STRUCTURAL VS SEMANTIC
# ============================================================

def create_structural_vs_semantic(df):

    required = [
        "length_similarity",
        "semantic_similarity"
    ]

    if not all(column in df.columns for column in required):
        print("Skipping structural vs semantic plot.")
        return

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=df,
        x="length_similarity",
        y="semantic_similarity",
        alpha=0.6
    )

    plt.title("Structural Similarity vs Semantic Similarity")
    plt.xlabel("Length Similarity")
    plt.ylabel("Semantic Similarity")
    plt.tight_layout()

    output = FIGURES_DIR / "structural_vs_semantic.png"
    plt.savefig(output, dpi=300)
    plt.close()

    print("Saved:", output)


# ============================================================
# QUALITY BANDS
# ============================================================

def add_quality_band(df):

    if "final_quality_score" not in df.columns:
        return df

    def classify(score):

        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Fair"
        else:
            return "Poor"

    df = df.copy()

    df["quality_band"] = (
        df["final_quality_score"]
        .apply(classify)
    )

    return df


def create_quality_band_chart(df):

    if "quality_band" not in df.columns:
        return

    counts = (
        df["quality_band"]
        .value_counts()
        .reindex(
            ["Excellent", "Good", "Fair", "Poor"],
            fill_value=0
        )
    )

    plt.figure(figsize=(9, 6))

    counts.plot(kind="bar")

    plt.title("MT Quality Band Distribution")
    plt.xlabel("Quality Band")
    plt.ylabel("Number of Sentences")
    plt.xticks(rotation=0)

    plt.tight_layout()

    output = FIGURES_DIR / "quality_band_distribution.png"
    plt.savefig(output, dpi=300)
    plt.close()

    print("Saved:", output)


# ============================================================
# SUMMARY TABLE
# ============================================================

def create_summary_table(df):

    columns = [
        "unigram_f1",
        "sentence_bleu",
        "edit_similarity",
        "length_similarity",
        "punctuation_similarity",
        "semantic_similarity",
        "lexical_quality_score",
        "structural_quality_score",
        "final_quality_score",
    ]

    available = [
        column for column in columns
        if column in df.columns
    ]

    if not available:
        return

    summary = df[available].describe().T

    output = TABLES_DIR / "score_summary.csv"
    summary.to_csv(output)

    print("Saved:", output)


# ============================================================
# GOOD / POOR EXAMPLES
# ============================================================

def create_quality_examples(df):

    if "final_quality_score" not in df.columns:
        return

    display_columns = [
        "idx",
        "source",
        "reference",
        "translation",
        "unigram_f1",
        "semantic_similarity",
        "length_similarity",
        "final_quality_score",
    ]

    available = [
        column for column in display_columns
        if column in df.columns
    ]

    sorted_df = df.sort_values(
        "final_quality_score",
        ascending=False
    )

    good_examples = sorted_df.head(20)[available]

    poor_examples = (
        sorted_df
        .tail(20)
        .sort_values("final_quality_score")
    )[available]

    good_file = TABLES_DIR / "good_examples.csv"
    poor_file = TABLES_DIR / "poor_examples.csv"

    good_examples.to_csv(
        good_file,
        index=False,
        encoding="utf-8-sig"
    )

    poor_examples.to_csv(
        poor_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("Saved:", good_file)
    print("Saved:", poor_file)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MT QUALITY EVALUATION - VISUALIZATION")
    print("=" * 70)

    df = load_data()

    df = add_quality_band(df)

    create_score_distributions(df)

    create_correlation_heatmap(df)

    create_lexical_vs_semantic(df)

    create_structural_vs_semantic(df)

    create_quality_band_chart(df)

    create_summary_table(df)

    create_quality_examples(df)

    print()
    print("=" * 70)
    print("VISUALIZATION COMPLETED")
    print("=" * 70)

    print("\nFigures saved in:")
    print(FIGURES_DIR)

    print("\nTables saved in:")
    print(TABLES_DIR)


if __name__ == "__main__":
    main()