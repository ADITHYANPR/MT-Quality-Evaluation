import os

os.environ["HF_HUB_OFFLINE"] = "1"
import time
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"

SOURCE_LANGUAGE = "eng_Latn"
TARGET_LANGUAGE = "mal_Mlym"


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    local_files_only=True
)

print("Loading model...")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    local_files_only=True
)

model.eval()

print("Model loaded successfully.")


# --------------------------------------------------
# Test sentences
# --------------------------------------------------

sentences = [
    "What has happened?",
    "They didn't talk.",
    "But India won.",
    "I can go.",
    "The weather is very pleasant today.",
    "Many people were arrested.",
    "Where is the railway station?",
    "This is an important decision.",
    "The project was completed successfully.",
    "We need to find a better solution."
]


# --------------------------------------------------
# Prepare input
# --------------------------------------------------

inputs = [
    f"{SOURCE_LANGUAGE} {sentence}"
    for sentence in sentences
]


print("\nStarting translation...\n")

start_time = time.time()

encoded = tokenizer(
    inputs,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=256
)

with torch.no_grad():

    generated_tokens = model.generate(
        **encoded,
        max_length=256,
        num_beams=5
    )


translations = tokenizer.batch_decode(
    generated_tokens,
    skip_special_tokens=True
)

elapsed = time.time() - start_time


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("=" * 70)
print("ENGLISH → MALAYALAM TEST")
print("=" * 70)

for i, (source, translation) in enumerate(
    zip(sentences, translations),
    start=1
):

    print(f"\n{i}. English:")
    print(source)

    print("Malayalam:")
    print(translation)


print("\n" + "=" * 70)
print(f"Translation time: {elapsed:.2f} seconds")
print("=" * 70)