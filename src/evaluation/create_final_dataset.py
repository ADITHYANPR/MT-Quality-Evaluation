import os
import pandas as pd

INPUT_FILE = "data/processed/experimental/samanantar_en_ml_semantic_1k.parquet"
OUTPUT_FILE = "data/processed/experimental/samanantar_en_ml_final_1k.parquet"


def main():

    print("=" * 70)
    print("CREATING FINAL MT EVALUATION DATASET")
    print("=" * 70)

    print()
    print("Loading semantic evaluation dataset...")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print()
    print("Checking missing values...")

    missing = df.isna().sum()

    if missing.sum() == 0:
        print("No missing values found.")
    else:
        print(missing[missing > 0])

    print()
    print("Checking duplicate rows...")

    duplicates = df.duplicated().sum()

    print(f"Duplicate rows: {duplicates}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("FINAL DATASET CREATED")
    print("=" * 70)

    print()
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print()
    print("Output:")
    print(OUTPUT_FILE)

    print()
    print("Semantic features:")
    print(" - semantic_cosine_similarity")
    print(" - semantic_similarity")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()