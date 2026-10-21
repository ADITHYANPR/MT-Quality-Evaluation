import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

INPUT_FILE = "data/processed/experimental/samanantar_en_ml_evaluation_1k.parquet"
OUTPUT_FILE = "data/processed/experimental/samanantar_en_ml_semantic_1k.parquet"

MODEL_NAME = "intfloat/multilingual-e5-small"
BATCH_SIZE = 8


def load_dataset():
    print("Loading dataset...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"Rows: {len(df)}")
    return df


def load_model():
    print("Loading semantic model...")
    print(f"Model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Semantic model loaded successfully.")
    return model


def calculate_semantic_similarity(model, references, translations):
    print("Encoding reference translations...")

    reference_embeddings = model.encode(
        references,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    print("Encoding MT translations...")

    translation_embeddings = model.encode(
        translations,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    similarities = np.sum(
        reference_embeddings * translation_embeddings,
        axis=1
    )

    similarities = np.clip(similarities, -1.0, 1.0)

    # Convert cosine similarity from [-1, 1] to [0, 1]
    normalized_similarity = (similarities + 1.0) / 2.0

    return similarities, normalized_similarity


def main():

    print("=" * 70)
    print("SEMANTIC MT QUALITY EVALUATION")
    print("=" * 70)

    df = load_dataset()

    print()
    model = load_model()

    references = df["reference"].fillna("").astype(str).tolist()
    translations = df["translation"].fillna("").astype(str).tolist()

    print()
    print("Calculating semantic similarity...")

    raw_similarity, normalized_similarity = calculate_semantic_similarity(
        model,
        references,
        translations
    )

    df["semantic_cosine_similarity"] = raw_similarity
    df["semantic_similarity"] = normalized_similarity

    print()
    print("=" * 70)
    print("SEMANTIC EVALUATION COMPLETED")
    print("=" * 70)

    print()
    print("Average semantic cosine similarity:")
    print(f"{df['semantic_cosine_similarity'].mean():.4f}")

    print()
    print("Average normalized semantic similarity:")
    print(f"{df['semantic_similarity'].mean():.4f}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("Output file:")
    print(OUTPUT_FILE)

    print()
    print("New semantic features:")
    print(" - semantic_cosine_similarity")
    print(" - semantic_similarity")

    print()
    print("=" * 70)
    print("SAMPLE SEMANTIC EVALUATIONS")
    print("=" * 70)

    for _, row in df.head(5).iterrows():

        print()
        print("-" * 70)

        print("REFERENCE:")
        print(row["reference"])

        print()

        print("MT OUTPUT:")
        print(row["translation"])

        print()

        print(
            f"Cosine Similarity : "
            f"{row['semantic_cosine_similarity']:.4f}"
        )

        print(
            f"Semantic Similarity : "
            f"{row['semantic_similarity']:.4f}"
        )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()