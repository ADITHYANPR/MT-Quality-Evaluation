from pathlib import Path
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = PROCESSED_DIR / "experimental"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Load cleaned shards
# --------------------------------------------------

files = sorted(
    PROCESSED_DIR.glob(
        "train-*-of-00004-clean.parquet"
    )
)

print(f"Found {len(files)} cleaned shards.")


dfs = []

for file in files:

    print(f"Loading: {file.name}")

    temp = pd.read_parquet(file)

    dfs.append(temp)


clean_df = pd.concat(
    dfs,
    ignore_index=True
)

print(
    f"\nTotal clean pairs: "
    f"{len(clean_df):,}"
)


# --------------------------------------------------
# Keep only English sources with ONE translation
# --------------------------------------------------

translation_counts = (
    clean_df.groupby("src")["tgt"]
    .nunique()
)

single_reference_sources = (
    translation_counts[
        translation_counts == 1
    ].index
)

single_ref_df = clean_df[
    clean_df["src"].isin(
        single_reference_sources
    )
].copy()

print(
    f"Pairs with a single "
    f"Malayalam reference: "
    f"{len(single_ref_df):,}"
)


# --------------------------------------------------
# Remove duplicate source rows
# --------------------------------------------------

single_ref_df = (
    single_ref_df
    .drop_duplicates(subset=["src"])
    .reset_index(drop=True)
)


print(
    f"Unique one-to-one pairs: "
    f"{len(single_ref_df):,}"
)


# --------------------------------------------------
# Randomly select 100,000 pairs
# --------------------------------------------------

SAMPLE_SIZE = 100_000

if len(single_ref_df) < SAMPLE_SIZE:

    raise ValueError(
        "Not enough one-to-one pairs "
        "to create the 100K dataset."
    )


experimental_df = single_ref_df.sample(
    n=SAMPLE_SIZE,
    random_state=42
).reset_index(drop=True)


# --------------------------------------------------
# Add project-friendly column names
# --------------------------------------------------

experimental_df = experimental_df.rename(
    columns={
        "src": "source",
        "tgt": "reference"
    }
)


# --------------------------------------------------
# Save
# --------------------------------------------------

output_file = (
    OUTPUT_DIR /
    "samanantar_en_ml_100k.parquet"
)

experimental_df.to_parquet(
    output_file,
    index=False
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("EXPERIMENTAL DATASET CREATED")
print("=" * 60)

print(
    f"\nDataset size : "
    f"{len(experimental_df):,}"
)

print(
    f"Output file  : "
    f"{output_file}"
)

print("\nColumns:")
print(experimental_df.columns.tolist())

print("\nSample:")
print(experimental_df.head())

print("=" * 60)