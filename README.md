# Supervised Learning: Comparison of Five Classification Algorithms

## Project Overview
This project compares five supervised machine-learning classification algorithms on the same real-world dataset. All models use the same training/testing split and are evaluated with the same metrics to make the comparison fair.

## Dataset
**Breast Cancer Wisconsin Diagnostic Dataset**, loaded from `sklearn.datasets`.

- 569 observations
- 30 numerical input features
- Two target classes: malignant and benign
- No categorical input columns, so categorical encoding is not needed.

The dataset is included with scikit-learn; no separate download is required.

## Algorithms
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Support Vector Machine (SVM) with RBF kernel
5. Gaussian Naive Bayes

## Preprocessing and Experimental Setup
- Split: 80% training and 20% testing.
- `random_state=42` and stratification are used to make the split reproducible and preserve class proportions.
- Median imputation is included in every model pipeline. The selected dataset has no missing values, but this step makes the workflow robust if missing values are introduced.
- Standard scaling is applied to Logistic Regression, KNN, and SVM because these algorithms are sensitive to feature magnitudes.
- Decision Trees and Gaussian Naive Bayes are not scaled because scaling is not generally required for these models.
- The scaler and imputer are fitted only on training data through scikit-learn pipelines, preventing data leakage.
- All models are trained and tested using the same split.

## Evaluation Metrics
- **Accuracy:** proportion of all predictions that are correct.
- **Precision:** proportion of positive predictions that are correct.
- **Recall:** proportion of actual positive cases correctly identified.
- **F1-score:** harmonic mean of precision and recall.

The positive class in scikit-learn's dataset is `malignant` (target value 0) only if labels are recoded; in this project the original scikit-learn target encoding is retained (`0 = malignant`, `1 = benign`). Scikit-learn's binary precision/recall/F1 defaults use class 1 (`benign`) as the positive class. The classification reports show metrics for both classes. For medical interpretation, review malignant-class recall as well.

## How to Run

```bash
pip install -r requirements.txt
python supervised_learning.py
```

The script creates:
- `model_comparison.csv` — ranked metrics table
- `model_comparison.png` — accuracy comparison chart
- `classification_reports.txt` — detailed per-class metrics

## Results

Run the script to generate the actual measured results. The ranked table is saved to `model_comparison.csv` and printed in the terminal. Do not substitute estimated or manually entered scores.

## Analysis

The best model is the one that achieves the highest test accuracy in the generated comparison table. The RBF SVM may perform well because it can model non-linear decision boundaries, while scaling keeps the 30 features on comparable ranges. Logistic Regression can also perform strongly when the classes are reasonably separable by a linear boundary. KNN depends on meaningful distances, which is why scaling is important. A single Decision Tree can be sensitive to small changes in training data and may overfit. Gaussian Naive Bayes assumes conditional independence between features; measurements describing cell nuclei can be correlated, so this assumption may limit its performance.

The actual best and lowest-performing models can vary with the split and implementation. Use the generated results to name the winner and lowest scorer, then relate those measured outcomes to the characteristics above.

## Limitations
This is an educational comparison on one dataset and one fixed split. It is not a clinical diagnostic system. Accuracy alone is not sufficient for medical decisions; class-specific recall, precision, and external validation are important.


## Actual Results From This Run

| Rank | Algorithm | Accuracy | Precision | Recall | F1-score |
|---:|---|---:|---:|---:|---:|
| 1 | Logistic Regression | 0.9825 | 0.9861 | 0.9861 | 0.9861 |
| 2 | Support Vector Machine | 0.9825 | 0.9861 | 0.9861 | 0.9861 |
| 3 | K-Nearest Neighbors | 0.9561 | 0.9589 | 0.9722 | 0.9655 |
| 4 | Gaussian Naive Bayes | 0.9386 | 0.9452 | 0.9583 | 0.9517 |
| 5 | Decision Tree | 0.9123 | 0.9559 | 0.9028 | 0.9286 |

**Best by accuracy:** Logistic Regression (0.9825). **Lowest by accuracy:** Decision Tree (0.9123).

On this fixed 80/20 split, Logistic Regression achieved the highest accuracy. Its performance is consistent with its ability to capture the decision structure present in these 30 numerical features. Decision Tree had the lowest accuracy in this run. A single tree can be sensitive to the particular training sample and may overfit.