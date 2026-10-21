import time
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Helsinki-NLP/opus-mt-en-ml"

INPUT_FILE = (
    "data/processed/experimental/"
    "samanantar_en_ml_100k.parquet"
)

OUTPUT_FILE = (
    "data/processed/experimental/"
    "samanantar_en_ml_mt_5k.parquet"
)

SAMPLE_SIZE = 1000
BATCH_SIZE = 8


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("ENGLISH → MALAYALAM MT OUTPUT GENERATION")
print("=" * 70)

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

model.eval()

device = torch.device("cpu")
model.to(device)

print("Model loaded successfully.")
print("Device:", device)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading experimental dataset...")

df = pd.read_parquet(INPUT_FILE)

print("Total available rows:", len(df))

if len(df) < SAMPLE_SIZE:
    raise ValueError(
        f"Dataset contains only {len(df)} rows, "
        f"but {SAMPLE_SIZE} were requested."
    )

# Fixed random_state makes the experiment reproducible
df = df.sample(
    n=SAMPLE_SIZE,
    random_state=42
).reset_index(drop=True)

print("Selected rows:", len(df))


# ============================================================
# TRANSLATION
# ============================================================

translations = []

print("\nStarting translation...")
print("Batch size:", BATCH_SIZE)

start_time = time.time()

for start in range(0, len(df), BATCH_SIZE):

    batch = df["source"].iloc[
        start:start + BATCH_SIZE
    ].tolist()

    inputs = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            num_beams=4,
            max_new_tokens=128
        )

    batch_translations = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )

    translations.extend(batch_translations)

    completed = min(
        start + BATCH_SIZE,
        len(df)
    )

    if completed % 100 == 0 or completed == len(df):
        elapsed = time.time() - start_time

        print(
            f"Translated {completed}/{len(df)} "
            f"({completed / len(df) * 100:.1f}%) "
            f"| Time: {elapsed:.1f}s"
        )


# ============================================================
# ADD TRANSLATIONS
# ============================================================

df["translation"] = translations


# ============================================================
# SAVE RESULT
# ============================================================

df.to_parquet(
    OUTPUT_FILE,
    index=False
)

elapsed = time.time() - start_time


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MT OUTPUT GENERATION COMPLETED")
print("=" * 70)

print("Rows generated :", len(df))
print("Output file    :", OUTPUT_FILE)
print(f"Time taken     : {elapsed / 60:.2f} minutes")

print("\nColumns:")
print(df.columns.tolist())

print("\nSample results:")

for i in range(min(5, len(df))):

    print("\n" + "-" * 70)

    print("SOURCE:")
    print(df.iloc[i]["source"])

    print("\nREFERENCE:")
    print(df.iloc[i]["reference"])

    print("\nMT OUTPUT:")
    print(df.iloc[i]["translation"])

print("\n" + "=" * 70)