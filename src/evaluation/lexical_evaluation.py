import pandas as pd
import numpy as np

from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.metrics import edit_distance


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    "data/processed/experimental/"
    "samanantar_en_ml_mt_1k.parquet"
)

OUTPUT_FILE = (
    "data/processed/experimental/"
    "samanantar_en_ml_lexical_1k.parquet"
)


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):
    """
    Simple whitespace-based tokenization.
    Malayalam words remain as Unicode text.
    """
    return str(text).strip().split()


# ============================================================
# UNIGRAM PRECISION
# ============================================================

def unigram_precision(reference, translation):

    reference_words = tokenize(reference)
    translation_words = tokenize(translation)

    if len(translation_words) == 0:
        return 0.0

    reference_counts = Counter(reference_words)
    translation_counts = Counter(translation_words)

    matched = 0

    for word, count in translation_counts.items():
        matched += min(
            count,
            reference_counts.get(word, 0)
        )

    return matched / len(translation_words)


# ============================================================
# UNIGRAM RECALL
# ============================================================

def unigram_recall(reference, translation):

    reference_words = tokenize(reference)
    translation_words = tokenize(translation)

    if len(reference_words) == 0:
        return 0.0

    reference_counts = Counter(reference_words)
    translation_counts = Counter(translation_words)

    matched = 0

    for word, count in reference_counts.items():
        matched += min(
            count,
            translation_counts.get(word, 0)
        )

    return matched / len(reference_words)


# ============================================================
# F1 SCORE
# ============================================================

def unigram_f1(precision, recall):

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        / (precision + recall)
    )


# ============================================================
# SENTENCE BLEU
# ============================================================

def calculate_bleu(reference, translation):

    reference_words = tokenize(reference)
    translation_words = tokenize(translation)

    if len(translation_words) == 0:
        return 0.0

    smoothing = SmoothingFunction().method1

    score = sentence_bleu(
        [reference_words],
        translation_words,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing
    )

    return score


# ============================================================
# EDIT DISTANCE
# ============================================================

def calculate_edit_distance(reference, translation):

    reference_words = tokenize(reference)
    translation_words = tokenize(translation)

    return edit_distance(
        reference_words,
        translation_words
    )


# ============================================================
# NORMALIZED EDIT SIMILARITY
# ============================================================

def normalized_edit_similarity(reference, translation):

    reference_words = tokenize(reference)
    translation_words = tokenize(translation)

    max_length = max(
        len(reference_words),
        len(translation_words)
    )

    if max_length == 0:
        return 1.0

    distance = edit_distance(
        reference_words,
        translation_words
    )

    similarity = 1 - (
        distance / max_length
    )

    return max(0.0, similarity)


# ============================================================
# LEXICAL EVALUATION
# ============================================================

def evaluate_row(row):

    reference = row["reference"]
    translation = row["translation"]

    precision = unigram_precision(
        reference,
        translation
    )

    recall = unigram_recall(
        reference,
        translation
    )

    f1 = unigram_f1(
        precision,
        recall
    )

    bleu = calculate_bleu(
        reference,
        translation
    )

    edit_dist = calculate_edit_distance(
        reference,
        translation
    )

    edit_similarity = normalized_edit_similarity(
        reference,
        translation
    )

    return pd.Series({
        "unigram_precision": precision,
        "unigram_recall": recall,
        "unigram_f1": f1,
        "sentence_bleu": bleu,
        "edit_distance": edit_dist,
        "edit_similarity": edit_similarity
    })


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("LEXICAL MT QUALITY EVALUATION")
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


# ------------------------------------------------------------
# Calculate lexical features
# ------------------------------------------------------------

print("\nCalculating lexical metrics...")

metrics = df.apply(
    evaluate_row,
    axis=1
)

df = pd.concat(
    [df, metrics],
    axis=1
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

df.to_parquet(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("LEXICAL EVALUATION COMPLETED")
print("=" * 70)

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nMetrics generated:")

for column in metrics.columns:
    print(" -", column)


print("\nAverage scores:")

for column in metrics.columns:
    print(
        f"{column:25s}: "
        f"{df[column].mean():.4f}"
    )


# ------------------------------------------------------------
# Show examples
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE EVALUATIONS")
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

    print("\nLEXICAL SCORES:")

    print(
        f"Unigram Precision : "
        f"{row['unigram_precision']:.4f}"
    )

    print(
        f"Unigram Recall    : "
        f"{row['unigram_recall']:.4f}"
    )

    print(
        f"Unigram F1        : "
        f"{row['unigram_f1']:.4f}"
    )

    print(
        f"Sentence BLEU     : "
        f"{row['sentence_bleu']:.4f}"
    )

    print(
        f"Edit Distance     : "
        f"{row['edit_distance']}"
    )

    print(
        f"Edit Similarity   : "
        f"{row['edit_similarity']:.4f}"
    )

print("\n" + "=" * 70)