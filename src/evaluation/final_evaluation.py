from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

# final_evaluation.py is inside:
# MT_Quality_Evaluation/src/evaluation/
#
# parents[0] = evaluation
# parents[1] = src
# parents[2] = MT_Quality_Evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "experimental"
    / "samanantar_en_ml_final_1k.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "experimental"
    / "samanantar_en_ml_quality_scored_1k.parquet"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "idx",
    "source",
    "reference",
    "translation",

    "unigram_precision",
    "unigram_recall",
    "unigram_f1",
    "sentence_bleu",
    "edit_distance",
    "edit_similarity",

    "source_word_count",
    "reference_word_count",
    "translation_word_count",
    "length_ratio",
    "length_difference",
    "length_similarity",
    "punctuation_count_difference",
    "punctuation_similarity",
    "translation_repetition_ratio",
    "maximum_word_repetition",

    "semantic_cosine_similarity",
    "semantic_similarity",
]


# ============================================================
# HELPERS
# ============================================================

def clip_01(series):
    """
    Convert values to numeric and restrict them to [0, 1].
    """

    return (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .fillna(0.0)
        .clip(0.0, 1.0)
    )


# ============================================================
# LEXICAL QUALITY
# ============================================================

def calculate_lexical_quality(df):
    """
    Calculate lexical quality score.

    Weighting:
        Unigram F1       = 40%
        Sentence BLEU    = 20%
        Edit Similarity  = 40%

    Output:
        Score between 0 and 1
    """

    unigram_f1 = clip_01(
        df["unigram_f1"]
    )

    sentence_bleu = clip_01(
        df["sentence_bleu"]
    )

    edit_similarity = clip_01(
        df["edit_similarity"]
    )

    lexical_score = (
        0.40 * unigram_f1
        + 0.20 * sentence_bleu
        + 0.40 * edit_similarity
    )

    return lexical_score.clip(
        0.0,
        1.0
    )


# ============================================================
# STRUCTURAL QUALITY
# ============================================================

def calculate_structural_quality(df):
    """
    Calculate structural quality score.

    Weighting:
        Length similarity              = 65%
        Punctuation similarity         = 15%
        Repetition quality             = 10%
        Maximum repetition quality     = 10%

    Output:
        Score between 0 and 1
    """

    length_similarity = clip_01(
        df["length_similarity"]
    )

    punctuation_similarity = clip_01(
        df["punctuation_similarity"]
    )

    repetition_ratio = (
        pd.to_numeric(
            df["translation_repetition_ratio"],
            errors="coerce"
        )
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    repetition_quality = (
        1.0 - repetition_ratio
    )

    maximum_repetition = (
        pd.to_numeric(
            df["maximum_word_repetition"],
            errors="coerce"
        )
        .fillna(1.0)
        .clip(lower=1.0)
    )

    maximum_repetition_quality = (
        1.0 / maximum_repetition
    ).clip(0.0, 1.0)

    structural_score = (
        0.65 * length_similarity
        + 0.15 * punctuation_similarity
        + 0.10 * repetition_quality
        + 0.10 * maximum_repetition_quality
    )

    return structural_score.clip(
        0.0,
        1.0
    )


# ============================================================
# FINAL QUALITY SCORE
# ============================================================

def calculate_final_quality_score(df):
    """
    Calculate final machine translation quality score.

    Weighting:
        Semantic quality   = 50%
        Lexical quality    = 30%
        Structural quality = 20%

    Final output:
        Score between 0 and 100
    """

    lexical_score = clip_01(
        df["lexical_quality_score"]
    )

    structural_score = clip_01(
        df["structural_quality_score"]
    )

    semantic_score = clip_01(
        df["semantic_similarity"]
    )

    final_score = (
        0.30 * lexical_score
        + 0.20 * structural_score
        + 0.50 * semantic_score
    )

    return (
        final_score
        .clip(0.0, 1.0)
        * 100
    )


# ============================================================
# QUALITY BAND
# ============================================================

def assign_quality_band(score):
    """
    Convert final score into a quality category.
    """

    if score >= 80:
        return "Excellent"

    elif score >= 60:
        return "Good"

    elif score >= 40:
        return "Fair"

    else:
        return "Poor"


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(df):
    """
    Validate input dataset before scoring.
    """

    print()
    print("=" * 70)
    print("DATASET VALIDATION")
    print("=" * 70)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    print("Required columns: OK")

    important_scores = [
        "unigram_f1",
        "sentence_bleu",
        "edit_similarity",
        "length_similarity",
        "punctuation_similarity",
        "semantic_similarity"
    ]

    for column in important_scores:

        missing = df[column].isna().sum()

        print(
            f"{column}: "
            f"{missing} missing values"
        )

        if missing > 0:
            raise ValueError(
                f"{column} contains missing values."
            )


# ============================================================
# REMOVE OLD SCORE COLUMNS
# ============================================================

def remove_existing_score_columns(df):
    """
    Remove previously generated scoring columns.

    This allows the script to be safely executed again.
    """

    score_columns = [
        "lexical_quality_score",
        "structural_quality_score",
        "final_quality_score",
        "quality_band"
    ]

    for column in score_columns:

        if column in df.columns:

            print(
                f"Removing existing column: {column}"
            )

            df = df.drop(
                columns=[column]
            )

    return df


# ============================================================
# VALIDATE FINAL SCORES
# ============================================================

def validate_final_scores(df):
    """
    Validate generated final quality scores.
    """

    if df["final_quality_score"].isna().any():

        raise ValueError(
            "Final quality score contains NaN values."
        )

    if (df["final_quality_score"] < 0).any():

        raise ValueError(
            "Final quality score contains values below 0."
        )

    if (df["final_quality_score"] > 100).any():

        raise ValueError(
            "Final quality score contains values above 100."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("FINAL MT QUALITY EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Display project paths
    # --------------------------------------------------------

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Input file:")
    print(INPUT_FILE)

    print()
    print("Output file:")
    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # Check input file
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        print()
        print("ERROR: Input file does not exist.")
        print()
        print("Expected file:")
        print(INPUT_FILE)
        print()

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    print()
    print("Input file found: OK")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print()
    print("Loading dataset...")

    df = pd.read_parquet(
        INPUT_FILE
    )

    print()
    print("Original shape:")
    print(df.shape)

    # --------------------------------------------------------
    # Validate dataset
    # --------------------------------------------------------

    validate_dataset(df)

    # --------------------------------------------------------
    # Remove old score columns
    # --------------------------------------------------------

    df = remove_existing_score_columns(
        df
    )

    # --------------------------------------------------------
    # Calculate lexical quality
    # --------------------------------------------------------

    print()
    print("Calculating lexical quality score...")

    df["lexical_quality_score"] = (
        calculate_lexical_quality(df)
    )

    # --------------------------------------------------------
    # Calculate structural quality
    # --------------------------------------------------------

    print()
    print("Calculating structural quality score...")

    df["structural_quality_score"] = (
        calculate_structural_quality(df)
    )

    # --------------------------------------------------------
    # Calculate final quality score
    # --------------------------------------------------------

    print()
    print("Calculating final quality score...")

    df["final_quality_score"] = (
        calculate_final_quality_score(df)
    )

    # --------------------------------------------------------
    # Assign quality band
    # --------------------------------------------------------

    print()
    print("Assigning quality bands...")

    df["quality_band"] = (
        df["final_quality_score"]
        .apply(assign_quality_band)
    )

    # --------------------------------------------------------
    # Validate final scores
    # --------------------------------------------------------

    print()
    print("Validating final scores...")

    validate_final_scores(df)

    print("Final scores: OK")

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    print()
    print("Saving final dataset...")

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL EVALUATION COMPLETED")
    print("=" * 70)

    print()
    print("Output:")
    print(OUTPUT_FILE)

    print()
    print("Final shape:")
    print(df.shape)

    print()
    print("New scoring columns:")

    print("  lexical_quality_score")
    print("  structural_quality_score")
    print("  final_quality_score")
    print("  quality_band")

    print()
    print("Score statistics:")

    print(
        df[
            [
                "lexical_quality_score",
                "structural_quality_score",
                "semantic_similarity",
                "final_quality_score"
            ]
        ].describe()
    )

    print()
    print("Quality bands:")

    print(
        df["quality_band"]
        .value_counts()
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()