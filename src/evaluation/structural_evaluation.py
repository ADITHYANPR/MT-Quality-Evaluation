import pandas as pd
import numpy as np
import re


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    "data/processed/experimental/"
    "samanantar_en_ml_lexical_1k.parquet"
)

OUTPUT_FILE = (
    "data/processed/experimental/"
    "samanantar_en_ml_structural_1k.parquet"
)


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):
    """Simple whitespace tokenization."""
    return str(text).strip().split()


# ============================================================
# WORD COUNT
# ============================================================

def word_count(text):
    return len(tokenize(text))


# ============================================================
# LENGTH RATIO
# ============================================================

def calculate_length_ratio(reference, translation):

    ref_count = word_count(reference)
    mt_count = word_count(translation)

    if ref_count == 0:
        return 0.0

    return mt_count / ref_count


# ============================================================
# LENGTH DIFFERENCE
# ============================================================

def calculate_length_difference(reference, translation):

    ref_count = word_count(reference)
    mt_count = word_count(translation)

    return abs(mt_count - ref_count)


# ============================================================
# LENGTH SIMILARITY
# ============================================================

def calculate_length_similarity(reference, translation):

    ref_count = word_count(reference)
    mt_count = word_count(translation)

    max_length = max(ref_count, mt_count)

    if max_length == 0:
        return 1.0

    difference = abs(ref_count - mt_count)

    return max(
        0.0,
        1 - (difference / max_length)
    )


# ============================================================
# PUNCTUATION EXTRACTION
# ============================================================

def extract_punctuation(text):

    return re.findall(
        r"[^\w\s]",
        str(text),
        flags=re.UNICODE
    )


# ============================================================
# PUNCTUATION COUNT DIFFERENCE
# ============================================================

def punctuation_count_difference(reference, translation):

    ref_punctuation = extract_punctuation(reference)
    mt_punctuation = extract_punctuation(translation)

    return abs(
        len(ref_punctuation)
        - len(mt_punctuation)
    )


# ============================================================
# PUNCTUATION SIMILARITY
# ============================================================

def punctuation_similarity(reference, translation):

    ref_punctuation = extract_punctuation(reference)
    mt_punctuation = extract_punctuation(translation)

    if len(ref_punctuation) == 0 and len(mt_punctuation) == 0:
        return 1.0

    max_length = max(
        len(ref_punctuation),
        len(mt_punctuation)
    )

    if max_length == 0:
        return 1.0

    matches = sum(
        1
        for a, b in zip(
            ref_punctuation,
            mt_punctuation
        )
        if a == b
    )

    return matches / max_length


# ============================================================
# REPETITION RATIO
# ============================================================

def repetition_ratio(text):

    words = tokenize(text)

    if len(words) == 0:
        return 0.0

    unique_words = set(words)

    repeated_words = len(words) - len(unique_words)

    return repeated_words / len(words)


# ============================================================
# MAXIMUM WORD REPETITION
# ============================================================

def maximum_word_repetition(text):

    words = tokenize(text)

    if len(words) == 0:
        return 0

    counts = {}

    for word in words:
        counts[word] = counts.get(word, 0) + 1

    return max(counts.values())


# ============================================================
# STRUCTURAL EVALUATION
# ============================================================

def evaluate_row(row):

    source = row["source"]
    reference = row["reference"]
    translation = row["translation"]

    source_words = word_count(source)
    reference_words = word_count(reference)
    translation_words = word_count(translation)

    length_ratio = calculate_length_ratio(
        reference,
        translation
    )

    length_difference = calculate_length_difference(
        reference,
        translation
    )

    length_similarity = calculate_length_similarity(
        reference,
        translation
    )

    punctuation_difference = punctuation_count_difference(
        reference,
        translation
    )

    punctuation_score = punctuation_similarity(
        reference,
        translation
    )

    mt_repetition = repetition_ratio(
        translation
    )

    max_repetition = maximum_word_repetition(
        translation
    )

    return pd.Series({

        "source_word_count": source_words,

        "reference_word_count": reference_words,

        "translation_word_count": translation_words,

        "length_ratio": length_ratio,

        "length_difference": length_difference,

        "length_similarity": length_similarity,

        "punctuation_count_difference":
            punctuation_difference,

        "punctuation_similarity":
            punctuation_score,

        "translation_repetition_ratio":
            mt_repetition,

        "maximum_word_repetition":
            max_repetition
    })


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("STRUCTURAL MT QUALITY EVALUATION")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_parquet(INPUT_FILE)

print("Rows:", len(df))


required_columns = [
    "idx",
    "source",
    "reference",
    "translation"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ============================================================
# CALCULATE STRUCTURAL FEATURES
# ============================================================

print("\nCalculating structural metrics...")

metrics = df.apply(
    evaluate_row,
    axis=1
)

df = pd.concat(
    [df, metrics],
    axis=1
)


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
print("STRUCTURAL EVALUATION COMPLETED")
print("=" * 70)

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nMetrics generated:")

for column in metrics.columns:

    print(" -", column)


print("\nAverage structural values:")

for column in metrics.columns:

    print(
        f"{column:35s}: "
        f"{df[column].mean():.4f}"
    )


# ============================================================
# SAMPLE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE STRUCTURAL EVALUATIONS")
print("=" * 70)

for i in range(min(5, len(df))):

    row = df.iloc[i]

    print("\n" + "-" * 70)

    print("SOURCE:")
    print(row["source"])

    print("\nREFERENCE:")
    print(row["reference"])

    print("\nMT OUTPUT:")
    print(row["translation"])

    print("\nSTRUCTURAL SCORES:")

    print(
        f"Source word count       : "
        f"{row['source_word_count']}"
    )

    print(
        f"Reference word count    : "
        f"{row['reference_word_count']}"
    )

    print(
        f"Translation word count  : "
        f"{row['translation_word_count']}"
    )

    print(
        f"Length ratio            : "
        f"{row['length_ratio']:.4f}"
    )

    print(
        f"Length difference       : "
        f"{row['length_difference']}"
    )

    print(
        f"Length similarity       : "
        f"{row['length_similarity']:.4f}"
    )

    print(
        f"Punctuation difference  : "
        f"{row['punctuation_count_difference']}"
    )

    print(
        f"Punctuation similarity  : "
        f"{row['punctuation_similarity']:.4f}"
    )

    print(
        f"Repetition ratio        : "
        f"{row['translation_repetition_ratio']:.4f}"
    )

    print(
        f"Maximum word repetition : "
        f"{row['maximum_word_repetition']}"
    )


print("\n" + "=" * 70)