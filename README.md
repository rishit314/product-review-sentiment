# Product Review Classification Using Sentiment Analysis and NLP Models

A machine-learning project that classifies Amazon product reviews into three sentiment categories: **Positive, Negative, and Neutral**. The project compares traditional TF-IDF-based classifiers with BERT-based models and demonstrates sentiment prediction through a Streamlit web application.

## Project Overview

Customer reviews contain useful information about product quality and customer experience. Automatically classifying review sentiment can help organize this feedback and identify general patterns in customer opinions.

This project investigates five classifiers and examines the effect of class weighting on performance, particularly for the underrepresented Neutral class.

## Models Evaluated

### Traditional machine-learning models
- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)

These models use Term Frequency–Inverse Document Frequency (TF-IDF) features.

### Transformer-based models
- Baseline BERT (`bert-base-uncased`)
- Class-weighted BERT

The class-weighted model assigns greater importance to underrepresented classes during training.

## Dataset

The project uses an Amazon product-review dataset from Zenodo:

- **Dataset record:** https://zenodo.org/records/3549704
- **DOI:** https://doi.org/10.5281/zenodo.3549704
- **Original dataset size:** 60,889 reviews
- **Reviews after cleaning:** 60,857

The data includes product categories, review headers, review text, ratings, and an existing sentiment label (`Own_Rating`).

The target labels are rating-derived Positive, Neutral, and Negative categories. They should not be interpreted as independent human annotations of the review text.

### Dataset split

| Subset | Reviews |
|---|---:|
| Training | 42,599 |
| Validation | 9,129 |
| Final test | 9,129 |
| **Total** | **60,857** |

The classes are imbalanced, with Positive reviews accounting for approximately 77.9% of each split.

## Final-Test Results

| Model | Accuracy | Macro-F1 | Weighted-F1 | Neutral-F1 |
|---|---:|---:|---:|---:|
| Naive Bayes | 81.56% | 0.4330 | 0.7577 | 0.0000 |
| Logistic Regression | 81.53% | 0.6335 | 0.8323 | 0.2686 |
| Linear SVM | 86.11% | 0.6352 | 0.8542 | 0.2281 |
| BERT — Baseline | **89.56%** | 0.6504 | **0.8761** | 0.1876 |
| BERT — Class-weighted | 85.57% | **0.7083** | 0.8706 | **0.3993** |

### Key observations

- Baseline BERT achieved the highest accuracy and weighted-F1.
- Class-weighted BERT achieved the highest Macro-F1 and Neutral-F1.
- Class weighting increased Neutral recall from approximately 11.8% to 55.3%, while reducing overall accuracy.
- Logistic Regression achieved higher Neutral-F1 than baseline BERT, despite being a traditional machine-learning model.
- Naive Bayes did not correctly identify any Neutral reviews.

These results show why accuracy alone is insufficient for evaluating classifiers on imbalanced datasets.

## Streamlit Application

The web application accepts a product review and predicts its sentiment using the saved TF-IDF vectorizer and Logistic Regression classifier.

It displays:
- The predicted sentiment
- Model-estimated class probabilities
- Input validation and error messages

The deployed application uses Logistic Regression, not BERT. Its prediction probabilities are model estimates and are not guaranteed to be calibrated.

### Run locally

1. Clone or download the repository.
2. Open a terminal in the project directory.
3. Create and activate a virtual environment.
4. Install the dependencies.
5. Run Streamlit.

On Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```text
product-review-sentiment/
├── data/
│   ├── raw/
│   └── processed/
│       └── final_split/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
├── src/
├── models/
│   └── final_split/
├── reports/
│   ├── experiment_summary.json
│   ├── final_model_comparison.csv
│   ├── traditional_final_test_results.csv
│   └── results_and_discussion.md
├── app.py
├── requirements.txt
└── README.md
```

Dataset files and trained model artifacts may be excluded from version control because of their size. The required files must be available locally for the application to run.

## Technologies

- Python
- pandas and NumPy
- scikit-learn
- Hugging Face Transformers
- PyTorch
- Streamlit
- joblib
- Jupyter Notebook

## Limitations

1. The sentiment labels are derived from the source dataset's rating-based labels, rather than independently annotated sentiment.
2. The dataset is imbalanced, especially for the Neutral class.
3. The final-test split overlaps with a test set examined earlier during development. Consequently, it cannot be described as a completely untouched holdout.
4. The models have been evaluated on this dataset and experimental setup; their performance may differ on other datasets, product categories, or real-world reviews.
5. The experiments do not establish statistical significance for the differences between models.

## Future Work

- Evaluate the models on a genuinely untouched external dataset.
- Investigate performance across product categories.
- Explore approaches that improve Neutral precision while maintaining Neutral recall.
- Evaluate probability calibration and prediction confidence.
- Improve the web application with batch prediction and review analytics.

## Project Title

**Product Review Classification Using Sentiment Analysis and NLP Models**
