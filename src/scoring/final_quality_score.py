import os
import pandas as pd
import numpy as np

INPUT_FILE = "data/processed/experimental/samanantar_en_ml_final_1k.parquet"
OUTPUT_FILE = "data/processed/experimental/samanantar_en_ml_quality_1k.parquet"


def clip(value, low=0.0, high=1.0):
    return np.clip(value, low, high)


def calculate_scores(df):

    # -------------------------------
    # LEXICAL SCORE
    # -------------------------------

    lexical_score = (
        0.40 * df["unigram_f1"] +
        0.30 * df["sentence_bleu"] +
        0.30 * df["edit_similarity"]
    )

    lexical_score = clip(lexical_score)

    # -------------------------------
    # STRUCTURAL SCORE
    # -------------------------------

    repetition_score = 1 - clip(
        df["translation_repetition_ratio"]
    )

    structural_score = (
        0.55 * df["length_similarity"] +
        0.25 * df["punctuation_similarity"] +
        0.20 * repetition_score
    )

    structural_score = clip(structural_score)

    # -------------------------------
    # SEMANTIC SCORE
    # -------------------------------

    semantic_score = clip(
        df["semantic_similarity"]
    )

    # -------------------------------
    # FINAL SCORE
    # -------------------------------

    final_score = (
        0.25 * lexical_score +
        0.20 * structural_score +
        0.55 * semantic_score
    )

    final_score = clip(final_score)

    df["lexical_quality_score"] = lexical_score
    df["structural_quality_score"] = structural_score
    df["semantic_quality_score"] = semantic_score

    df["final_quality_score"] = final_score
    df["final_quality_percentage"] = final_score * 100

    # Quality category

    conditions = [
        df["final_quality_percentage"] >= 80,
        df["final_quality_percentage"] >= 60,
        df["final_quality_percentage"] >= 40
    ]

    choices = [
        "Excellent",
        "Good",
        "Moderate"
    ]

    df["quality_category"] = np.select(
        conditions,
        choices,
        default="Poor"
    )

    return df


def main():

    print("=" * 70)
    print("FINAL MT QUALITY SCORING")
    print("=" * 70)

    print()
    print("Loading dataset...")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Rows: {len(df)}")

    print()
    print("Calculating quality scores...")

    df = calculate_scores(df)

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("FINAL QUALITY SCORING COMPLETED")
    print("=" * 70)

    print()
    print("Average scores:")

    print(
        f"Lexical Quality     : "
        f"{df['lexical_quality_score'].mean():.4f}"
    )

    print(
        f"Structural Quality  : "
        f"{df['structural_quality_score'].mean():.4f}"
    )

    print(
        f"Semantic Quality    : "
        f"{df['semantic_quality_score'].mean():.4f}"
    )

    print(
        f"Final Quality       : "
        f"{df['final_quality_percentage'].mean():.2f}/100"
    )

    print()
    print("Quality distribution:")

    print(
        df["quality_category"]
        .value_counts()
    )

    print()
    print("Output:")
    print(OUTPUT_FILE)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()