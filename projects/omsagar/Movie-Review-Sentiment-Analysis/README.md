# 🎬 Movie Review Sentiment Analysis

A machine learning system that classifies IMDB movie reviews as Positive or Negative, built as part of an AI/ML internship (following earlier House Price Prediction and Diabetes Prediction projects).

## Overview

*   **Type:** Binary text classification
*   **Dataset:** IMDB 50K Movie Reviews (Kaggle)
*   **Goal:** Predict sentiment (Positive/Negative) from raw review text

## Pipeline

### 1\. Text Preprocessing

*   HTML tag removal, punctuation/number stripping, lowercasing, stopword removal
*   Lemmatization chosen over stemming — produces real dictionary words, better feature quality

### 2\. Feature Extraction

Compared Bag of Words vs TF-IDF:

| Representation | Accuracy |
| --- | --- |
| Bag of Words | 87.02% |
| TF-IDF | 88.83% |

TF-IDF selected for all further modeling.

### 3\. Model Comparison

| Model | Accuracy | Train Time |
| --- | --- | --- |
| Logistic Regression | 88.83% | 1.87s |
| Linear SVM | 88.35% | 1.83s |
| Naive Bayes | 85.31% | 0.04s |
| Random Forest | 84.59% | 52.38s |

### 4\. Tuning

Added bigrams (captures phrases like "not good") + GridSearchCV hyperparameter tuning (best `C=5`) → final accuracy **89.59%**.

## Explainability

Used LIME (Local Interpretable Model-agnostic Explanations) to show which words drive each prediction, making the model interpretable rather than a black box.

## Error Analysis

10.41% misclassification rate. Common failure patterns identified:

*   **Comparison reviews** — reviewer criticizes one thing (e.g. a remake) while praising another (e.g. the original), confusing the bag-of-words representation
*   **Mid-review sentiment shifts** — reviews that start positive and turn negative (or vice versa); TF-IDF has no concept of word order
*   **Backhanded compliments / sarcasm** — positive words used in a critical context
*   **Heavy negative vocabulary describing content, not opinion** — e.g. reviews discussing a film's dark subject matter while rating it positively

These are known limitations of linear bag-of-words models and motivate exploring sequence-aware models.

## Deep Learning Comparison (LSTM)

Trained an LSTM (Embedding + LSTM(64) + Dense) on the same dataset for comparison.

| Model | Test Accuracy |
| --- | --- |
| Logistic Regression (TF-IDF + bigrams, tuned) | 89.59% |
| LSTM | 86.41% |

**Finding:** The LSTM overfit (training accuracy reached 97.7%, validation plateaued at ~86%), while the simpler TF-IDF + Logistic Regression model generalized better. This highlights that on moderately-sized text datasets, well-tuned classical ML can outperform deep learning without heavy regularization or more data — a known and reportable result, not a failure.

**Final model chosen: Logistic Regression (TF-IDF, bigrams, tuned)** — best accuracy, fastest inference, easiest to deploy and explain.

## Model Versioning

Trained models are saved with date-stamped versions in `models/versions/` alongside the production copy in `models/`, so past versions can be compared or rolled back.

## Demo App (Streamlit)

Interactive UI with:

*   Single review prediction with LIME word-contribution visualization
*   Batch CSV analysis with sentiment distribution chart

Run locally:  
streamlit run app/app.py

## API (FastAPI)

A REST API wraps the trained model for production use.

*   `GET /` — health check
*   `POST /predict` — accepts `{"text": "..."}`, returns `{"sentiment": "...", "confidence": ...}`
*   Interactive docs auto-generated at `/docs` (Swagger UI)

Run locally:

cd api  
uvicorn main:app --reload

## Docker

The API is fully containerized — runs anywhere with no manual Python setup.  
docker build -t sentiment-api -f api/Dockerfile .  
docker run -d -p 8000:8000 --name sentiment-container sentiment-api

Then visit `http://localhost:8000/docs` to test.

## Tech Stack

Python, scikit-learn, NLTK, TensorFlow/Keras, LIME, FastAPI, Docker, Streamlit, pandas, matplotlib

## Project Structure

Movie-Review-Sentiment-Analysis/  
├── api/  
│ ├── main.py  
│ ├── Dockerfile  
│ └── requirements.txt  
├── app/  
│ └── app.py  
├── data/  
│ └── IMDB Dataset.csv  
├── models/  
│ ├── sentiment\_model.pkl  
│ ├── tfidf\_vectorizer.pkl  
│ └── versions/  
├── notebooks/  
│ └── sentiment\_analysis.ipynb  
├── images/  
└── README.md

## Full Pipeline Summary

Data cleaning → lemmatization → TF-IDF (bigrams) → Logistic Regression (tuned, GridSearchCV) → LIME explainability → error analysis → LSTM comparison → FastAPI → Docker → Streamlit demo