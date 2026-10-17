# Results and Discussion

## 1. Dataset and Experimental Setup

The experiments used a dataset of Amazon product reviews containing three sentiment classes: Negative, Neutral, and Positive. After preprocessing, 60,857 reviews were divided into training, validation, and final-test subsets using stratified splitting.

| Subset | Number of reviews |
|---|---:|
| Training | 42,599 |
| Validation | 9,129 |
| Final test | 9,129 |
| **Total** | **60,857** |

The training subset contained 33,185 Positive reviews (77.90%), 6,360 Negative reviews (14.93%), and 3,054 Neutral reviews (7.17%). The validation and final-test subsets had nearly identical class proportions.

Three traditional machine-learning classifiers—Multinomial Naive Bayes, Logistic Regression, and Linear Support Vector Machine (SVM)—were evaluated using TF-IDF text features. Two BERT-based classifiers were also evaluated: a baseline BERT model and a class-weighted BERT model designed to place greater emphasis on underrepresented classes.

Performance was measured using accuracy, macro-averaged F1-score, weighted F1-score, and class-specific F1-scores. Macro-F1 is particularly relevant because it gives equal weight to each class, whereas accuracy and weighted-F1 can be strongly influenced by the dominant Positive class.

## 2. Overall Model Comparison

| Model | Accuracy | Macro-F1 | Weighted-F1 | Negative F1 | Neutral F1 | Positive F1 |
|---|---:|---:|---:|---:|---:|---:|
| Naive Bayes | 0.8156 | 0.4330 | 0.7577 | 0.4036 | 0.0000 | 0.8954 |
| Logistic Regression | 0.8153 | 0.6335 | 0.8323 | 0.7274 | 0.2686 | 0.9044 |
| Linear SVM | 0.8611 | 0.6352 | 0.8542 | 0.7447 | 0.2281 | 0.9328 |
| BERT — Baseline | **0.8956** | 0.6504 | **0.8761** | **0.8116** | 0.1876 | **0.9519** |
| BERT — Class-weighted | 0.8557 | **0.7083** | 0.8706 | 0.7975 | **0.3993** | 0.9280 |

Baseline BERT achieved the highest accuracy (89.56%) and weighted-F1 (0.8761). The class-weighted BERT model achieved the highest macro-F1 (0.7083) and Neutral-class F1 (0.3993).

Among the traditional classifiers, Linear SVM achieved the highest accuracy (86.11%) and weighted-F1 (0.8542). Logistic Regression obtained a higher Neutral-class F1 (0.2686) than either baseline BERT or Linear SVM. Naive Bayes failed to predict any Neutral reviews correctly, resulting in a Neutral F1-score of zero.

These results demonstrate that the preferred model depends on the evaluation objective. Baseline BERT is strongest when overall accuracy is prioritized, while class-weighted BERT offers more balanced performance across the three sentiment classes.

## 3. Effect of Class Weighting

Class weighting changed the distribution of predictions made by BERT. Compared with baseline BERT, class-weighted BERT increased macro-F1 from 0.6504 to 0.7083, an improvement of 0.0579. Neutral F1 increased from 0.1876 to 0.3993, an improvement of 0.2117.

Neutral recall increased from approximately 11.8% to 55.3%. The number of correctly identified Neutral reviews increased from 77 to 362 out of 655 actual Neutral reviews.

This improvement came with a reduction in accuracy from 89.56% to 85.57%, and a reduction in Positive F1 from 0.9519 to 0.9280. Neutral precision also remained limited: approximately 46% for baseline BERT and 31% for class-weighted BERT.

The confusion matrices indicate that class weighting reduces the number of Neutral reviews incorrectly classified as Positive, but it also increases false Neutral predictions among Negative and Positive reviews. Thus, class weighting improves the detection of a minority class without eliminating the underlying classification trade-off.

## 4. Discussion

The results highlight the limitations of relying exclusively on accuracy for imbalanced sentiment classification. Since approximately 77.9% of the reviews are labeled Positive, a model can achieve high accuracy while performing poorly on Neutral reviews.

The Naive Bayes model illustrates this issue clearly. Despite achieving 81.56% accuracy, it predicted no Neutral reviews. Logistic Regression and Linear SVM provided stronger minority-class recognition, although their Neutral-class performance remained limited.

Baseline BERT achieved the strongest overall accuracy and weighted-F1, suggesting effective classification of the dominant class and strong performance across the dataset. However, its Neutral F1-score was only 0.1876. The class-weighted BERT model improved macro-F1 and Neutral F1 substantially, indicating that class weighting can be useful when minority-class detection is an important objective.

The findings therefore support evaluating sentiment classifiers using multiple complementary metrics. In practical applications, model selection should depend on whether the cost of missing Neutral reviews or incorrectly labeling other reviews as Neutral is more important.

## 5. Limitations

Several limitations should be acknowledged.

First, the target sentiment labels are derived from the dataset's existing rating-based `Own_Rating` field. They should not be interpreted as independently human-annotated sentiment labels unless that annotation process is separately established.

Second, the classes are substantially imbalanced, with Positive reviews forming approximately 77.9% of the dataset. Performance may differ on datasets with different class distributions or product categories.

Third, the designated final-test split overlaps with a test set examined earlier during development. Although the current split is internally disjoint and its labels were checked against the source dataset by review ID, previous exposure means it cannot be described as a completely untouched holdout. This limits the strength of claims about generalization.

Fourth, the experiments do not establish that the observed differences are statistically significant. Repeated runs, confidence intervals, or an appropriate paired comparison would be needed to quantify uncertainty in the metric differences.

Finally, the evaluation covers the specific models, preprocessing choices, and experimental settings used in this project. The results should not be generalized to all sentiment-analysis systems or all product-review datasets.

## 6. Conclusion

The experiments show that model performance depends on the metric used to evaluate it. Baseline BERT achieved the highest accuracy and weighted-F1, whereas class-weighted BERT achieved the highest macro-F1 and Neutral-class F1. Among traditional models, Linear SVM provided the strongest overall performance by accuracy and weighted-F1, while Logistic Regression performed better on Neutral reviews.

The central finding is that class weighting improved the recognition of underrepresented Neutral reviews and increased balanced classification performance, but reduced overall accuracy. This demonstrates the importance of considering class-level metrics when developing sentiment-classification systems for imbalanced review datasets.

Further work should evaluate the models on a genuinely untouched dataset or holdout, investigate category-wise performance, and assess whether improved minority-class detection can be achieved while controlling false Neutral predictions.
