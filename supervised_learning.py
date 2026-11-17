"""
Supervised Learning: Comparison of Five Classification Algorithms
Dataset: Breast Cancer Wisconsin Diagnostic Dataset (scikit-learn)
Run: python supervised_learning.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

OUTPUT_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42

def main():
    # 1. Load a real dataset
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target

    # 2. Use one fixed train/test split for every model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # 3. Define five different supervised learning algorithms.
    # Imputation is included consistently in every pipeline.
    # Scaling is used for models sensitive to feature scale.
    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE))
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Support Vector Machine": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", random_state=RANDOM_STATE))
        ]),
        "Gaussian Naive Bayes": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GaussianNB())
        ])
    }

    # 4. Train and evaluate all models on the identical test set
    rows = []
    reports = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        metrics = {
            "Algorithm": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1-score": f1_score(y_test, predictions, zero_division=0)
        }
        rows.append(metrics)
        reports[name] = classification_report(
            y_test, predictions, target_names=dataset.target_names, zero_division=0
        )

    results = pd.DataFrame(rows).sort_values(
        by=["Accuracy", "F1-score"], ascending=False
    ).reset_index(drop=True)
    results.insert(0, "Rank", range(1, len(results) + 1))

    # 5. Save the comparison table
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print("\nMODEL COMPARISON (sorted by accuracy)")
    print(results.to_string(index=False, formatters={
        col: (lambda value: f"{value:.4f}") for col in ["Accuracy", "Precision", "Recall", "F1-score"]
    }))

    # 6. Create one comparison chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results["Algorithm"], results["Accuracy"])
    plt.title("Accuracy Comparison of Five Supervised Learning Algorithms")
    plt.xlabel("Algorithm")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, results["Accuracy"]):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.012,
                 f"{value:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=200)
    plt.close()

    # 7. Save per-model classification reports
    with open(OUTPUT_DIR / "classification_reports.txt", "w", encoding="utf-8") as file:
        for name, report in reports.items():
            file.write(f"\n{'=' * 65}\n{name}\n{'=' * 65}\n{report}\n")

    # 8. Print dataset and split information
    print("\nDATASET INFORMATION")
    print(f"Dataset: {dataset.DESCR.splitlines()[0]}")
    print(f"Total samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Missing values before imputation: {int(X.isna().sum().sum())}")
    print("\nBEST MODEL:", results.iloc[0]["Algorithm"])
    print("LOWEST ACCURACY MODEL:", results.iloc[-1]["Algorithm"])
    print("\nFiles generated: model_comparison.csv, model_comparison.png, classification_reports.txt")

if __name__ == "__main__":
    main()
