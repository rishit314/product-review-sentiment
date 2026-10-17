
import time
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "shared_test.csv"
MODEL_DIR = ROOT / "models"

test_df = pd.read_csv(DATA_PATH)

vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")

# Verify which model is saved before running predictions.
model = joblib.load(MODEL_DIR / "sentiment_model.joblib")

X_test = vectorizer.transform(test_df["text"])
y_true = test_df["label"].to_numpy()

start = time.perf_counter()
y_pred = model.predict(X_test)
elapsed = time.perf_counter() - start

print("Model:", type(model).__name__)
print("Test examples:", len(test_df))
print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
print("Macro-F1:", round(
    f1_score(y_true, y_pred, average="macro"), 4
))
print("\nClassification report:")
print(classification_report(y_true, y_pred, digits=4))
print("Prediction time (seconds):", round(elapsed, 3))

# Save predictions for comparison with BERT.
output = test_df[["test_id", "label"]].copy()
output["traditional_prediction"] = y_pred
output.to_csv(
    ROOT / "data" / "processed" / "traditional_predictions.csv",
    index=False,
)
print("\nPredictions saved.")
