
from pathlib import Path

import pandas as pd


# Resolve the project root from this script's location.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "amazon_reviews.csv"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    print("=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)
    print(f"File: {DATA_PATH.name}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nCOLUMN NAMES")
    for column in df.columns:
        print(f"- {column}")

    print("\nDATA TYPES")
    print(df.dtypes.to_string())

    print("\nFIRST 5 ROWS")
    print(df.head().to_string(max_cols=10, max_colwidth=60))

    print("\nMISSING VALUES")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("No missing values detected.")
    else:
        print(missing.to_string())

    print(f"\nExact duplicate rows: {df.duplicated().sum():,}")

    # Identify likely rating columns without assuming a fixed name.
    rating_columns = [
        column for column in df.columns
        if any(term in column.lower() for term in ("rating", "stars", "score"))
    ]

    print("\nPOTENTIAL RATING COLUMNS")
    if not rating_columns:
        print("No obvious rating column found.")
    else:
        for column in rating_columns:
            print(f"\n{column}:")
            print(df[column].value_counts(dropna=False).head(15).to_string())
    print("\nRATING VS SENTIMENT LABEL VALIDATION")

    rating_to_sentiment = {
        1: "Negative",
        2: "Negative",
        3: "Neutral",
        4: "Positive",
        5: "Positive",
    }

    expected_labels = df["Rating"].map(rating_to_sentiment)
    supplied_labels = df["Own_Rating"].str.strip().str.title()

    mismatch_mask = (
        expected_labels.notna()
        & supplied_labels.notna()
        & expected_labels.ne(supplied_labels)
    )

    print(f"Rows with mismatched labels: {mismatch_mask.sum():,}")

    if mismatch_mask.any():
        print("\nExamples of mismatches:")
        print(
            df.loc[mismatch_mask, ["Rating", "Own_Rating"]]
            .head(10)
            .to_string(index=False)
        )

    print("\nEMPTY OR WHITESPACE-ONLY TEXT")
    for column in ["Review_Header", "Review_text"]:
        empty_mask = (
            df[column].isna()
            | df[column].astype("string").str.strip().eq("").fillna(False)
        )
        print(f"{column}: {empty_mask.sum():,} empty values")

    print("\nRATING-TO-SENTIMENT MAPPING")
    print(
        pd.DataFrame({
            "Rating": df["Rating"],
            "Own_Rating": supplied_labels,
            "Expected_Label": expected_labels,
        })
        .groupby(["Rating", "Own_Rating", "Expected_Label"], dropna=False)
        .size()
        .rename("Count")
        .reset_index()
        .to_string(index=False)
    )

    print("\nInspection complete. The original CSV was not modified.")


if __name__ == "__main__":
    main()
