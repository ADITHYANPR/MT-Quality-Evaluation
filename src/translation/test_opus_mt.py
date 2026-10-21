import time
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "Helsinki-NLP/opus-mt-en-ml"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

model.eval()

print("Model loaded successfully.")


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


print("\nStarting translation...\n")

start = time.time()

inputs = tokenizer(
    sentences,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=256
)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        num_beams=4,
        max_new_tokens=128
    )

translations = tokenizer.batch_decode(
    outputs,
    skip_special_tokens=True
)

elapsed = time.time() - start


print("=" * 70)
print("ENGLISH → MALAYALAM TEST")
print("=" * 70)

for i, (source, translation) in enumerate(
    zip(sentences, translations), 1
):
    print(f"\n{i}. English:")
    print(source)

    print("Malayalam:")
    print(translation)

print("\n" + "=" * 70)
print(f"Translation time: {elapsed:.2f} seconds")
print("=" * 70)