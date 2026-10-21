import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    "data/processed/experimental/"
    "samanantar_en_ml_structural_1k.parquet"
)

OUTPUT_FILE = (
    "data/processed/experimental/"
    "samanantar_en_ml_evaluation_1k.parquet"
)


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("CREATING COMBINED MT EVALUATION DATASET")
print("=" * 70)

print("\nLoading structural evaluation dataset...")

df = pd.read_parquet(INPUT_FILE)

print("Rows:", len(df))


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = [
    "idx",
    "source",
    "reference",
    "translation",

    # Lexical
    "unigram_precision",
    "unigram_recall",
    "unigram_f1",
    "sentence_bleu",
    "edit_distance",
    "edit_similarity",

    # Structural
    "source_word_count",
    "reference_word_count",
    "translation_word_count",
    "length_ratio",
    "length_difference",
    "length_similarity",
    "punctuation_count_difference",
    "punctuation_similarity",
    "translation_repetition_ratio",
    "maximum_word_repetition"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# SELECT FINAL COLUMNS
# ============================================================

df = df[required_columns]


# ============================================================
# CHECK DATA
# ============================================================

print("\nChecking missing values...")

missing = df.isna().sum()

print(missing[missing > 0])

if missing.sum() == 0:
    print("No missing values found.")


print("\nChecking duplicate rows...")

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)


# ============================================================
# SAVE
# ============================================================

df.to_parquet(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("COMBINED DATASET CREATED")
print("=" * 70)

print("\nRows:", len(df))

print("\nColumns:")

for column in df.columns:
    print(" -", column)

print("\nOutput file:")
print(OUTPUT_FILE)


# ============================================================
# GROUP SUMMARY
# ============================================================

lexical_columns = [
    "unigram_precision",
    "unigram_recall",
    "unigram_f1",
    "sentence_bleu",
    "edit_distance",
    "edit_similarity"
]

structural_columns = [
    "source_word_count",
    "reference_word_count",
    "translation_word_count",
    "length_ratio",
    "length_difference",
    "length_similarity",
    "punctuation_count_difference",
    "punctuation_similarity",
    "translation_repetition_ratio",
    "maximum_word_repetition"
]


print("\nLexical features:", len(lexical_columns))
print("Structural features:", len(structural_columns))

print("\nFinal dataset shape:")
print(df.shape)

print("\n" + "=" * 70)
print("READY FOR NEXT EVALUATION STAGE")
print("=" * 70)