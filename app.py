
from pathlib import Path

import joblib
import streamlit as st


# Locate the saved final-split model artifacts
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models" / "final_split"

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"


@st.cache_resource
def load_artifacts():
    """Load the fitted TF-IDF vectorizer and Logistic Regression model."""
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model


st.set_page_config(
    page_title="Product Review Sentiment Analyzer",
    page_icon="🛍️",
    layout="centered",
)

st.title("🛍️ Product Review Sentiment Analyzer")

st.write(
    "Enter a product review to predict whether its sentiment "
    "is Positive, Neutral, or Negative."
)

st.info(
    "This model was trained on Amazon product reviews with "
    "rating-derived sentiment labels. Predictions may be "
    "incorrect, especially for mixed or ambiguous reviews."
)

if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
    st.error(
        "Required model files were not found. Expected files:\n\n"
        f"- {VECTORIZER_PATH}\n"
        f"- {MODEL_PATH}"
    )
    st.stop()

try:
    vectorizer, model = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the model artifacts: {exc}")
    st.stop()

review_text = st.text_area(
    "Enter your product review",
    placeholder="Example: The product works well and is worth the money.",
    height=160,
)

if st.button("Analyze Sentiment", type="primary"):
    if not review_text.strip():
        st.warning("Please enter a review first.")
    else:
        try:
            features = vectorizer.transform([review_text.strip()])
            prediction = model.predict(features)[0]

            st.subheader("Prediction")

            if prediction == "Positive":
                st.success("Positive sentiment")
            elif prediction == "Negative":
                st.error("Negative sentiment")
            elif prediction == "Neutral":
                st.warning("Neutral sentiment")
            else:
                st.error(f"Unexpected model output: {prediction}")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]

                st.subheader("Model-estimated class probabilities")

                for class_name, probability in zip(
                    model.classes_, probabilities
                ):
                    st.write(f"**{class_name}:** {probability:.1%}")
                    st.progress(float(probability))

                st.caption(
                    "These are model-estimated probabilities, not "
                    "guaranteed measures of prediction correctness."
                )
            else:
                st.info(
                    "Probability estimates are not available for this model."
                )

        except Exception as exc:
            st.error(f"Could not analyze this review: {exc}")
