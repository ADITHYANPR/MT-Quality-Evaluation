import pandas as pd

FILE = "data/processed/experimental/samanantar_en_ml_quality_scored_1k.parquet"

df = pd.read_parquet(FILE)

L = df["lexical_quality_score"]
S = df["structural_quality_score"]
M = df["semantic_similarity"]

print("=" * 60)
print("FINAL SCORE VALIDATION")
print("=" * 60)

print("\nCurrent scores:")
print(f"Lexical mean    : {L.mean():.4f}")
print(f"Structural mean : {S.mean():.4f}")
print(f"Semantic mean   : {M.mean():.4f}")
print(f"Final mean      : {df['final_quality_score'].mean():.2f}")

print("\nScore ranges:")
print(f"Lexical    : {L.min():.4f} - {L.max():.4f}")
print(f"Structural : {S.min():.4f} - {S.max():.4f}")
print(f"Semantic   : {M.min():.4f} - {M.max():.4f}")
print(f"Final      : {df['final_quality_score'].min():.2f} - {df['final_quality_score'].max():.2f}")

print("\nCorrelations:")
print(df[
    [
        "lexical_quality_score",
        "structural_quality_score",
        "semantic_similarity",
        "final_quality_score"
    ]
].corr()["final_quality_score"].sort_values(ascending=False))

print("\nWeight sensitivity:")

weights = [
    ("Current", 0.30, 0.20, 0.50),
    ("Equal",   0.333, 0.333, 0.334),
    ("Lexical-heavy", 0.50, 0.20, 0.30),
    ("Semantic-heavy", 0.20, 0.20, 0.60),
]

for name, lw, sw, mw in weights:

    score = (
        lw * L +
        sw * S +
        mw * M
    ) * 100

    print(
        f"{name:18} "
        f"Mean={score.mean():6.2f} "
        f"Min={score.min():6.2f} "
        f"Max={score.max():6.2f}"
    )

    print(
        f"   Poor={sum(score < 40):4d}  "
        f"Fair={sum((score >= 40) & (score < 60)):4d}  "
        f"Good={sum((score >= 60) & (score < 80)):4d}  "
        f"Excellent={sum(score >= 80):4d}"
    )

print("\n" + "=" * 60)