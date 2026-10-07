"""
Trains a high-accuracy, regularized Sentiment Classification Ensemble
on data/training_data.csv and saves the fitted vectorizer and ensemble model
to model/vectorizer.pkl and model/sentiment_model.pkl.

Ensemble Architecture:
- Hybrid Word + Subword Character TF-IDF (FeatureUnion)
- Regularized Soft-Voting Ensemble:
  * Calibrated Linear Support Vector Classifier (LinearSVC)
  * Complement Naive Bayes (ComplementNB)
  * L2-Regularized Logistic Regression
  * Modified Huber SGD Classifier

Run this once (or whenever training_data.csv changes):
    python model/train_model.py
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")


def build_vectorizer():
    """
    Constructs a hybrid feature extraction pipeline combining word n-grams
    and character boundary n-grams to capture root semantics, prefixes/suffixes,
    misspellings, and punctuation markers without overfitting.
    """
    return FeatureUnion([
        (
            "word_tfidf",
            TfidfVectorizer(
                ngram_range=(1, 3),
                sublinear_tf=True,
                min_df=1,
                token_pattern=r"(?u)\b\w+\b|[!?:;]+|:\)|:\(|:D|<3"
            )
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                sublinear_tf=True,
                min_df=2
            )
        )
    ])


def build_model():
    """
    Constructs a soft-voting ensemble combining calibrated linear SVM,
    complement naive bayes, regularized logistic regression, and modified huber SGD.
    """
    svc = CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42), cv=3)
    cnb = ComplementNB(alpha=0.3)
    lr = LogisticRegression(C=3.0, max_iter=1000, random_state=42)
    sgd = SGDClassifier(loss="modified_huber", penalty="l2", alpha=1e-4, random_state=42)

    return VotingClassifier(
        estimators=[
            ("cnb", cnb),
            ("svc", svc),
            ("lr", lr),
            ("sgd", sgd)
        ],
        voting="soft",
        weights=[2.0, 2.0, 1.5, 1.0]
    )


def main():
    print("=" * 60)
    print("Training Enhanced Sentiment Analysis Model")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    df = df.dropna()
    print(f"Total training samples: {len(df)}")
    print("Class distribution:")
    for label, count in df["label"].value_counts().items():
        print(f"  - {label:<10}: {count} samples ({count / len(df):.1%})")

    X = df["text"]
    y = df["label"]

    # 1. Evaluate with 5-Fold Stratified Cross-Validation
    print("\n[1/3] Running 5-Fold Stratified Cross-Validation...")
    vectorizer_cv = build_vectorizer()
    X_vec_all = vectorizer_cv.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_cv = build_model()
    cv_scores = cross_val_score(model_cv, X_vec_all, y, cv=cv, scoring="accuracy")
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

    # 2. Holdout Test Set Evaluation
    print("\n[2/3] Evaluating on 20% Holdout Test Set...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = build_model()
    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    print(f"Holdout Test Accuracy: {acc:.2%}\n")
    print("Classification Report:")
    print(classification_report(y_test, preds, digits=4))

    # 3. Fit on full dataset and save production artifacts
    print("[3/3] Training final production model on full dataset...")
    final_vectorizer = build_vectorizer()
    X_full_vec = final_vectorizer.fit_transform(X)
    final_model = build_model()
    final_model.fit(X_full_vec, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(final_vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
    joblib.dump(final_model, os.path.join(MODEL_DIR, "sentiment_model.pkl"))

    print(f"\nSuccessfully saved model + vectorizer to {MODEL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
