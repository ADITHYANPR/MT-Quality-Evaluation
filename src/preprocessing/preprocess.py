import re
import pandas as pd


# --------------------------------------------------
# 1. Malayalam Script Check
# --------------------------------------------------

def get_malayalam_char_count(text):
    """Count Malayalam Unicode characters in a text."""
    if not isinstance(text, str):
        return 0

    return len(re.findall(r"[\u0D00-\u0D7F]", text))


# --------------------------------------------------
# 2. Latin Character Check
# --------------------------------------------------

def get_latin_char_count(text):
    """Count English/Latin alphabet characters in a text."""
    if not isinstance(text, str):
        return 0

    return len(re.findall(r"[A-Za-z]", text))


# --------------------------------------------------
# 3. Web / HTML Noise Detection
# --------------------------------------------------

def contains_web_noise(text):
    """Detect obvious URLs, HTML or website-related contamination."""
    if not isinstance(text, str):
        return True

    pattern = r"<[^>]+>|https?://|www\.|\.com\b|\.in\b"

    return bool(
        re.search(pattern, text, flags=re.IGNORECASE)
    )


# --------------------------------------------------
# 4. Basic Text Validation
# --------------------------------------------------

def is_valid_text(text):
    """Check whether text is non-empty and valid."""
    return (
        isinstance(text, str)
        and bool(text.strip())
    )


# --------------------------------------------------
# 5. Calculate Text Statistics
# --------------------------------------------------

def add_text_statistics(df):
    """Add sentence-level statistics."""

    df["src_words"] = (
        df["src"].str.split().str.len()
    )

    df["tgt_words"] = (
        df["tgt"].str.split().str.len()
    )

    df["malayalam_chars"] = (
        df["tgt"].apply(get_malayalam_char_count)
    )

    df["latin_chars"] = (
        df["tgt"].apply(get_latin_char_count)
    )

    total_relevant_chars = (
        df["malayalam_chars"] +
        df["latin_chars"]
    )

    df["latin_ratio"] = (
        df["latin_chars"] /
        total_relevant_chars.replace(0, 1)
    )

    df["length_ratio"] = (
        df["tgt_words"] /
        df["src_words"].replace(0, 1)
    )

    return df


# --------------------------------------------------
# 6. Quality Filtering
# --------------------------------------------------

def filter_dataset(df):
    """Apply conservative quality filters."""

    # Basic validation
    valid_basic = (
        df["src"].apply(is_valid_text) &
        df["tgt"].apply(is_valid_text)
    )

    # Remove obvious web/HTML contamination
    no_web_noise = (
        ~df["src"].apply(contains_web_noise) &
        ~df["tgt"].apply(contains_web_noise)
    )

    # Malayalam target should contain meaningful Malayalam
    reasonable_malayalam = (
        df["latin_ratio"] <= 0.50
    )

    # Remove extreme sentence lengths
    reasonable_length = (
        (df["src_words"] <= 100) &
        (df["tgt_words"] <= 100)
    )

    # Remove extreme source-target length mismatch
    reasonable_ratio = (
        (df["length_ratio"] >= 0.20) &
        (df["length_ratio"] <= 3.0)
    )

    final_mask = (
        valid_basic &
        no_web_noise &
        reasonable_malayalam &
        reasonable_length &
        reasonable_ratio
    )

    return df[final_mask].copy()


# --------------------------------------------------
# 7. Process One Parquet File
# --------------------------------------------------

def process_parquet_file(input_file, output_file):
    """Load, clean and save one Parquet shard."""

    print(f"\nProcessing: {input_file}")

    df = pd.read_parquet(input_file)

    original_size = len(df)

    # Add statistics
    df = add_text_statistics(df)

    # Apply filters
    df = filter_dataset(df)

    # Keep only required project columns
    df = df[
        ["idx", "src", "tgt"]
    ]

    # Save processed data
    df.to_parquet(
        output_file,
        index=False
    )

    print(f"Original rows : {original_size:,}")
    print(f"Clean rows    : {len(df):,}")
    print(
        f"Removed       : "
        f"{original_size - len(df):,}"
    )

    return df


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":
    print("Preprocessing module loaded successfully.")