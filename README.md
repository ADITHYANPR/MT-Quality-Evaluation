# Machine Translation Quality Evaluation for English-to-Malayalam

## Overview

This project implements a multi-dimensional framework for evaluating machine-generated English-to-Malayalam translations. It combines lexical, structural, and semantic features into a configurable composite quality score and provides an interactive Streamlit dashboard for exploring evaluation results.

## Objectives

- Preprocess the Samanantar English-Malayalam dataset.
- Generate translations using the OPUS-MT model.
- Calculate lexical evaluation metrics.
- Analyse structural characteristics of translations.
- Estimate semantic similarity using multilingual sentence embeddings.
- Combine evaluation dimensions into a composite score.
- Classify translations into quality bands.
- Visualize evaluation results through a dashboard.

## Methodology

1. Dataset preprocessing.
2. Translation generation using `Helsinki-NLP/opus-mt-en-ml`.
3. Lexical evaluation using unigram precision, recall, F1, sentence BLEU, and edit similarity.
4. Structural evaluation using sentence length, punctuation, and repetition features.
5. Semantic evaluation using `intfloat/multilingual-e5-small`.
6. Weighted score fusion and quality-band classification.
7. Statistical analysis, visualizations, and dashboard exploration.

## Experimental Results

The initial experiment evaluated 1,000 machine-generated translations.

- Mean composite score: 65.87/100.
- Good: 795 translations.
- Fair: 161 translations.
- Excellent: 44 translations.
- Poor: 0 translations.

These results use heuristic weights and thresholds. They are preliminary estimates, not validated measures of translation accuracy. Human evaluation and comparison against established baselines remain future work.

## Technology Stack

- Python
- PyTorch and Transformers
- Sentence Transformers
- pandas, NumPy, and PyArrow
- NLTK and scikit-learn
- Matplotlib and Seaborn
- Streamlit and Plotly

## Running the Dashboard

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard requires the locally generated evaluation dataset at the path expected by the application. Dataset files and model weights are not included in this repository.

## Repository Structure

- `src/` - preprocessing, translation, evaluation, scoring, and analysis modules.
- `dashboard/` - Streamlit dashboard.
- `notebooks/` - dataset exploration notebooks.
- `results/` - generated figures and analysis tables.
- `tests/` - project tests.
- `requirements.txt` - Python dependencies.

## Limitations

The current implementation uses heuristic score weights, a limited evaluation sample, and a single reference translation per example. The scoring framework has not yet been validated against human judgments.

## Status

Experimental implementation completed; further validation and baseline benchmarking are planned.
