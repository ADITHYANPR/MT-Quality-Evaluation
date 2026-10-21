from pathlib import Path
import pandas as pd
from preprocess import process_parquet_file


# --------------------------------------------------
# Project Directories
# --------------------------------------------------

RAW_DIR = Path("data/raw/samanantar_ml")
PROCESSED_DIR = Path("data/processed")


# Create processed directory if it doesn't exist
PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Find all Parquet files
# --------------------------------------------------

input_files = sorted(
    RAW_DIR.glob("*.parquet")
)


# --------------------------------------------------
# Process all shards
# --------------------------------------------------

total_original = 0
total_clean = 0


print("=" * 60)
print("SAMANANTAR ENGLISH–MALAYALAM PREPROCESSING")
print("=" * 60)

print(f"\nFound {len(input_files)} Parquet files.\n")


for input_file in input_files:

    output_file = (
        PROCESSED_DIR /
        f"{input_file.stem}-clean.parquet"
    )

    df = process_parquet_file(
        input_file,
        output_file
    )

    original_rows = pd.read_parquet(
        input_file,
        columns=["idx"]
    ).shape[0]

    clean_rows = len(df)

    total_original += original_rows
    total_clean += clean_rows


# --------------------------------------------------
# Final Summary
# --------------------------------------------------

total_removed = (
    total_original - total_clean
)

retention = (
    total_clean / total_original * 100
)


print("\n")
print("=" * 60)
print("FINAL DATASET SUMMARY")
print("=" * 60)

print(
    f"\nTotal original pairs : "
    f"{total_original:,}"
)

print(
    f"Total clean pairs    : "
    f"{total_clean:,}"
)

print(
    f"Total removed        : "
    f"{total_removed:,}"
)

print(
    f"Retention            : "
    f"{retention:.2f}%"
)

print("=" * 60)
print("\nPreprocessing completed successfully!")