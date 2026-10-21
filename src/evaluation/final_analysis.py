import pandas as pd

INPUT_FILE = "data/processed/experimental/samanantar_en_ml_quality_scored_1k.parquet"


def main():

    print("=" * 70)
    print("FINAL MT QUALITY ANALYSIS")
    print("=" * 70)

    df = pd.read_parquet(INPUT_FILE)

    print(f"\nRows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nAverage quality scores:")

    score_columns = [
        "lexical_quality_score",
        "structural_quality_score",
        "semantic_similarity",
        "final_quality_score"
    ]

    print(df[score_columns].mean().to_string())

    print()
    print("=" * 70)
    print("CORRELATION WITH FINAL QUALITY")
    print("=" * 70)

    correlation_columns = [
        "unigram_f1",
        "sentence_bleu",
        "edit_similarity",
        "length_similarity",
        "punctuation_similarity",
        "translation_repetition_ratio",
        "semantic_similarity",
        "final_quality_score"
    ]

    print(
        df[correlation_columns]
        .corr()["final_quality_score"]
        .sort_values(ascending=False)
    )

    print()
    print("=" * 70)
    print("QUALITY BAND DISTRIBUTION")
    print("=" * 70)

    print(
        df["quality_band"]
        .value_counts()
        .reindex(
            ["Excellent", "Good", "Fair", "Poor"],
            fill_value=0
        )
    )

    print()
    print("=" * 70)
    print("BEST TRANSLATIONS")
    print("=" * 70)

    best = df.nlargest(5, "final_quality_score")

    for _, row in best.iterrows():

        print("\n" + "-" * 70)

        print("SOURCE:")
        print(row["source"])

        print("\nREFERENCE:")
        print(row["reference"])

        print("\nMT:")
        print(row["translation"])

        print(
            f"\nFinal Score: "
            f"{row['final_quality_score']:.2f}/100"
        )

    print()
    print("=" * 70)
    print("POOREST TRANSLATIONS")
    print("=" * 70)

    worst = df.nsmallest(5, "final_quality_score")

    for _, row in worst.iterrows():

        print("\n" + "-" * 70)

        print("SOURCE:")
        print(row["source"])

        print("\nREFERENCE:")
        print(row["reference"])

        print("\nMT:")
        print(row["translation"])

        print(
            f"\nFinal Score: "
            f"{row['final_quality_score']:.2f}/100"
        )


if __name__ == "__main__":
    main()