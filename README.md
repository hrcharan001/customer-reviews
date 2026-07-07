# 🛍️ CustomerSentiment-API

**End-to-end NLP pipeline that classifies customer reviews as Positive, Neutral, or Negative — deployed as a real-time REST API.**

---

## 📌 Overview

This project takes raw, messy customer review data and turns it into a deployable sentiment classification service. It covers the full ML lifecycle: data cleaning → text preprocessing → feature engineering → model benchmarking → API deployment.

## ⚙️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| NLP | NLTK (stopword removal, lemmatization) |
| Feature Engineering | TF-IDF Vectorizer (Scikit-learn) |
| Modeling | Logistic Regression, Naive Bayes, Linear SVM, XGBoost |
| Deployment | Flask, Pickle, Vercel |

## 🔄 Pipeline

1. **Data Cleaning** — handled missing values and duplicate reviews
2. **Text Preprocessing** — lowercasing, punctuation/URL/number removal, stopword removal, lemmatization
3. **Feature Engineering** — TF-IDF vectorization (5,000 max features, unigrams + bigrams)
4. **Model Benchmarking** — trained and compared 4 classification algorithms using precision, recall, and F1-score
5. **Deployment** — best model serialized with Pickle and exposed via a Flask REST API on Vercel

## 🚀 API Usage

**Endpoint:** `POST /predict`

**Request:**
```json
{
  "review": "The product quality is amazing, highly recommend!"
}
```

**Response:**
```json
{
  "status": "success",
  "review": "The product quality is amazing, highly recommend!",
  "predicted_sentiment": "Positive"
}
```

## 📂 Project Structure

```
├── Customer_reviews.ipynb    # EDA, preprocessing, model training & comparison
├── app.py                    # Flask REST API
├── sentiment_model.pkl       # Trained classification model
├── tfidf_vectorizer.pkl      # Fitted TF-IDF vectorizer
├── label_encoder.pkl         # Sentiment label encoder
├── requirements.txt          # Dependencies
└── vercel.json                # Deployment config
```

## 🔮 Future Improvements

- Scale to a larger, more diverse review dataset for stronger generalization
- Add confidence scores alongside predictions
- Build a lightweight frontend for live demo testing

---

**Author:** Hari Charan K | [GitHub](https://github.com/hrcharan001)
