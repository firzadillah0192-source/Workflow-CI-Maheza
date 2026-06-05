# ==========================================
# MODELLING.PY
# Dicoding - Membangun Sistem Machine Learning
# ==========================================

import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("=" * 50)
print("MEMULAI TRAINING MODEL")
print("=" * 50)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("dataset_preprocessing.csv")

df = df.dropna(subset=["clean_text", "target"])
df["clean_text"] = df["clean_text"].astype(str)

print(f"Jumlah Data : {len(df)}")

# ==========================================
# SPLIT DATA
# ==========================================

X = df["clean_text"]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# TF-IDF
# ==========================================

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.90,
    sublinear_tf=True
)

X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

print("Shape Train :", X_train.shape)
print("Shape Test  :", X_test.shape)

# ==========================================
# MLFLOW AUTOLOG
# ==========================================

mlflow.sklearn.autolog(
    log_models=True,
    silent=True
)

# ==========================================
# TRAIN MODEL
# ==========================================

model = LogisticRegression(
    max_iter=3000,
    C=2.0,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print()
print("=" * 50)
print(f"AKURASI MODEL : {accuracy:.4f}")
print("=" * 50)

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "tfidf.pkl")

print()
print("Model berhasil disimpan.")
print("TF-IDF berhasil disimpan.")

print()
print("=" * 50)
print("SELESAI")
print("=" * 50)