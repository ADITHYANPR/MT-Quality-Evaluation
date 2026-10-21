import re
from collections import Counter
from typing import List, Dict


def tokenize(text: str) -> List[str]:
    """
    Tokenize text using Unicode-aware word matching.

    Works for English and Malayalam text.
    """

    if text is None:
        return []

    text = str(text).strip()

    if not text:
        return []

    return re.findall(r"\w+", text, flags=re.UNICODE)


def unigram_precision(
    reference: str,
    translation: str
) -> float:
    """
    Calculate unigram precision using clipped counts.
    """

    reference_tokens = tokenize(reference)
    translation_tokens = tokenize(translation)

    if not translation_tokens:
        return 0.0

    reference_counts = Counter(reference_tokens)
    translation_counts = Counter(translation_tokens)

    matched = 0

    for token, count in translation_counts.items():
        matched += min(
            count,
            reference_counts.get(token, 0)
        )

    return matched / len(translation_tokens)


def unigram_recall(
    reference: str,
    translation: str
) -> float:
    """
    Calculate unigram recall using clipped counts.
    """

    reference_tokens = tokenize(reference)
    translation_tokens = tokenize(translation)

    if not reference_tokens:
        return 0.0

    reference_counts = Counter(reference_tokens)
    translation_counts = Counter(translation_tokens)

    matched = 0

    for token, count in reference_counts.items():
        matched += min(
            count,
            translation_counts.get(token, 0)
        )

    return matched / len(reference_tokens)


def unigram_f1(
    reference: str,
    translation: str
) -> float:
    """
    Calculate unigram F1 score.
    """

    precision = unigram_precision(
        reference,
        translation
    )

    recall = unigram_recall(
        reference,
        translation
    )

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        / (precision + recall)
    )


def edit_distance(
    reference: str,
    translation: str
) -> int:
    """
    Calculate character-level Levenshtein edit distance.
    """

    reference = "" if reference is None else str(reference)
    translation = "" if translation is None else str(translation)

    previous = list(range(len(translation) + 1))

    for i, ref_char in enumerate(reference, start=1):

        current = [i]

        for j, trans_char in enumerate(
            translation,
            start=1
        ):

            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1]

            if ref_char != trans_char:
                substitution += 1

            current.append(
                min(
                    insertion,
                    deletion,
                    substitution
                )
            )

        previous = current

    return previous[-1]


def edit_similarity(
    reference: str,
    translation: str
) -> float:
    """
    Convert edit distance into a normalized similarity score.
    """

    reference = "" if reference is None else str(reference)
    translation = "" if translation is None else str(translation)

    max_length = max(
        len(reference),
        len(translation)
    )

    if max_length == 0:
        return 1.0

    distance = edit_distance(
        reference,
        translation
    )

    return max(
        0.0,
        1.0 - distance / max_length
    )


def calculate_lexical_metrics(
    reference: str,
    translation: str
) -> Dict[str, float]:
    """
    Calculate all lexical metrics for one sentence pair.
    """

    precision = unigram_precision(
        reference,
        translation
    )

    recall = unigram_recall(
        reference,
        translation
    )

    f1 = unigram_f1(
        reference,
        translation
    )

    distance = edit_distance(
        reference,
        translation
    )

    similarity = edit_similarity(
        reference,
        translation
    )

    return {
        "unigram_precision": precision,
        "unigram_recall": recall,
        "unigram_f1": f1,
        "edit_distance": distance,
        "edit_similarity": similarity,
    }