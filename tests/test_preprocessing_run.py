from pathlib import Path

from src.preprocessing.preprocess import process_parquet_file


INPUT_FILE = Path(
    "data/raw/samanantar_ml/"
    "train-00000-of-00004.parquet"
)

OUTPUT_FILE = Path(
    "data/processed/"
    "train-00000-clean.parquet"
)


process_parquet_file(
    INPUT_FILE,
    OUTPUT_FILE
)