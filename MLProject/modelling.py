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
# MLFLOW
# ==========================================

mlflow.set_experiment("Sentiment_Analysis_Indonesia")

# Matikan autolog agar lebih stabil di GitHub Actions
# mlflow.sklearn.autolog()

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
# LOG KE MLFLOW
# ==========================================

mlflow.log_param("model", "LogisticRegression")
mlflow.log_param("max_iter", 3000)
mlflow.log_param("C", 2.0)

mlflow.log_metric("accuracy", accuracy)

mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model"
)

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