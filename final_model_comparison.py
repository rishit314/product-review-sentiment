from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"

FILES = {
    "Naive Bayes": ("naive_bayes_predictions.csv", "prediction"),
    "Linear SVM": ("linear_svm_predictions.csv", "prediction"),
    "Logistic Regression": ("traditional_predictions.csv", "traditional_prediction"),
    "BERT baseline": ("bert_baseline_predictions.csv", "prediction"),
    "BERT class-weighted": ("bert_weighted_predictions.csv", "prediction"),
}

shared = pd.read_csv(DATA / "shared_test.csv")
y_true = shared["label"].astype(str).to_numpy()
results = []

for model_name, (filename, pred_col) in FILES.items():
    path = DATA / filename
    if not path.exists():
        print(f"Skipping {model_name}: missing {filename}")
        continue

    pred_df = pd.read_csv(path)

    # Check row alignment before comparing predictions
    if len(pred_df) != len(shared):
        raise ValueError(f"{model_name}: row count mismatch")

    if "test_id" in pred_df.columns and not (
        pred_df["test_id"].to_numpy() == shared["test_id"].to_numpy()
    ).all():
        raise ValueError(f"{model_name}: test_id mismatch")

    if "label" in pred_df.columns and not (
        pred_df["label"].astype(str).to_numpy() == y_true
    ).all():
        raise ValueError(f"{model_name}: true labels mismatch")

    y_pred = pred_df[pred_col].astype(str).to_numpy()

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro-F1": f1_score(y_true, y_pred, average="macro"),
        "Neutral-F1": f1_score(
            y_true, y_pred, labels=["Neutral"],
            average="macro", zero_division=0
        ),
    })

summary = pd.DataFrame(results).sort_values("Macro-F1", ascending=False)
summary.to_csv(DATA / "final_model_comparison.csv", index=False)

print("\n=== Final model comparison ===")
print(summary.round(4).to_string(index=False))
print(f"\nSaved to: {DATA / 'final_model_comparison.csv'}")