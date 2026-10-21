import re
from collections import Counter
from typing import Dict


def tokenize(text: str):
    """
    Unicode-aware tokenization.
    """

    if text is None:
        return []

    text = str(text).strip()

    if not text:
        return []

    return re.findall(
        r"\w+",
        text,
        flags=re.UNICODE
    )


def word_count(text: str) -> int:
    """
    Count words in a sentence.
    """

    return len(
        tokenize(text)
    )


def punctuation_count(text: str) -> int:
    """
    Count punctuation characters.
    """

    if text is None:
        return 0

    return len(
        re.findall(
            r"[^\w\s]",
            str(text),
            flags=re.UNICODE
        )
    )


def length_ratio(
    reference: str,
    translation: str
) -> float:
    """
    Translation length / reference length.
    """

    reference_length = word_count(reference)
    translation_length = word_count(translation)

    if reference_length == 0:
        return 0.0

    return (
        translation_length
        / reference_length
    )


def length_difference(
    reference: str,
    translation: str
) -> int:
    """
    Absolute word-count difference.
    """

    return abs(
        word_count(reference)
        - word_count(translation)
    )


def length_similarity(
    reference: str,
    translation: str
) -> float:
    """
    Normalized length similarity.

    1.0 = same length
    0.0 = very different lengths
    """

    reference_length = word_count(reference)
    translation_length = word_count(translation)

    maximum_length = max(
        reference_length,
        translation_length
    )

    if maximum_length == 0:
        return 1.0

    difference = abs(
        reference_length
        - translation_length
    )

    return max(
        0.0,
        1.0 - difference / maximum_length
    )


def punctuation_count_difference(
    reference: str,
    translation: str
) -> int:
    """
    Absolute punctuation-count difference.
    """

    return abs(
        punctuation_count(reference)
        - punctuation_count(translation)
    )


def punctuation_similarity(
    reference: str,
    translation: str
) -> float:
    """
    Punctuation similarity.

    1.0 = identical punctuation count
    """

    reference_count = punctuation_count(
        reference
    )

    translation_count = punctuation_count(
        translation
    )

    maximum_count = max(
        reference_count,
        translation_count
    )

    if maximum_count == 0:
        return 1.0

    difference = abs(
        reference_count
        - translation_count
    )

    return max(
        0.0,
        1.0 - difference / maximum_count
    )


def repetition_ratio(
    text: str
) -> float:
    """
    Calculate proportion of repeated words.

    0.0 = no repetition
    """

    tokens = tokenize(text)

    if len(tokens) <= 1:
        return 0.0

    counts = Counter(tokens)

    repeated_tokens = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    return (
        repeated_tokens
        / len(tokens)
    )


def maximum_word_repetition(
    text: str
) -> int:
    """
    Maximum number of occurrences
    of a single word.
    """

    tokens = tokenize(text)

    if not tokens:
        return 0

    counts = Counter(tokens)

    return max(
        counts.values()
    )


def calculate_structural_metrics(
    source: str,
    reference: str,
    translation: str
) -> Dict[str, float]:
    """
    Calculate all structural metrics
    for one source-reference-translation pair.
    """

    return {
        "source_word_count":
            word_count(source),

        "reference_word_count":
            word_count(reference),

        "translation_word_count":
            word_count(translation),

        "length_ratio":
            length_ratio(
                reference,
                translation
            ),

        "length_difference":
            length_difference(
                reference,
                translation
            ),

        "length_similarity":
            length_similarity(
                reference,
                translation
            ),

        "punctuation_count_difference":
            punctuation_count_difference(
                reference,
                translation
            ),

        "punctuation_similarity":
            punctuation_similarity(
                reference,
                translation
            ),

        "translation_repetition_ratio":
            repetition_ratio(
                translation
            ),

        "maximum_word_repetition":
            maximum_word_repetition(
                translation
            ),
    }