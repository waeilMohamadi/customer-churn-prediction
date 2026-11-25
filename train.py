"""Train, evaluate, and register a customer churn model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL = ROOT / "models" / "churn_pipeline.joblib"
REPORT = ROOT / "metrics.json"

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
THRESHOLD = 0.5


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert blank or invalid TotalCharges values to numeric missing values."""
    frame = frame.copy()
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    return frame


def make_pipeline(features: list[str], estimator) -> Pipeline:
    """Keep preprocessing and the classifier together."""
    categorical = [column for column in features if column not in NUMERIC]

    # Learn numeric medians and scaling statistics from training rows.
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    # Fill missing categories and encode each category as an indicator.
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, categorical),
    ])

    # During cross-validation, each fold fits its own preprocessing.
    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", estimator),
    ])


def train() -> dict:
    # 1. Read the CSV and separate input features from the churn label.
    frame = prepare_frame(pd.read_csv(DATA))
    X = frame.drop(columns=["customerID", "Churn"])
    y = frame["Churn"].map({"No": 0, "Yes": 1})
    features = list(X.columns)

    # 2. Set aside 20% of customers for final testing.
    # Stratification preserves roughly the same churn rate in each split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # 3. Define the two candidate classifiers.
    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=15,
            learning_rate=0.08,
            random_state=RANDOM_STATE,
        ),
    }
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # 4. Compare candidates using training-only cross-validation.
    scores = {}
    for name, estimator in candidates.items():
        pipeline = make_pipeline(features, estimator)
        fold_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="average_precision",
        )
        scores[name] = {
            "mean_average_precision": round(float(fold_scores.mean()), 4),
            "std_average_precision": round(float(fold_scores.std()), 4),
        }

    # 5. Refit the winning pipeline on all training rows.
    selected = max(
        scores,
        key=lambda name: scores[name]["mean_average_precision"],
    )
    model = make_pipeline(features, candidates[selected])
    model.fit(X_train, y_train)

    # 6. Evaluate on the held-out test set.
    # Ranking metrics use probabilities. The report uses thresholded labels.
    churn_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X_test)[:, churn_index]
    predictions = (probabilities >= THRESHOLD).astype(int)

    report = {
        "dataset_rows": len(frame),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "selection_metric": "5-fold train-only average precision",
        "cross_validation": scores,
        "selected_model": selected,
        "held_out_test": {
            "average_precision": round(
                float(average_precision_score(y_test, probabilities)), 4
            ),
            "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
            "classification_threshold": THRESHOLD,
            "classification_report": classification_report(
                y_test, predictions, output_dict=True
            ),
        },
    }

    # 7. Save the local files for inspection and prediction comparisons.
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "features": features}, MODEL)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # 8. Record the training settings, evaluation metrics, and files in MLflow.
    mlflow.set_experiment("customer_churn")

    with mlflow.start_run(run_name=selected):
        mlflow.log_params({
            "selected_model": selected,
            "dataset_rows": len(frame),
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
            "random_state": RANDOM_STATE,
            "selection_metric": "average_precision",
            "classification_threshold": THRESHOLD,
        })

        for name, result in scores.items():
            mlflow.log_metric(
                f"cv_{name}_average_precision",
                result["mean_average_precision"],
            )

        test = report["held_out_test"]
        churn = test["classification_report"]["1"]
        mlflow.log_metrics({
            "test_roc_auc": test["roc_auc"],
            "test_average_precision": test["average_precision"],
            "test_churn_precision": churn["precision"],
            "test_churn_recall": churn["recall"],
            "test_churn_f1": churn["f1-score"],
        })

        mlflow.log_artifact(str(REPORT))
        mlflow.log_artifact(str(MODEL), artifact_path="model")

        # 9. Register the fitted pipeline so FastAPI can load it by version.
        # These specific Skops types are used by your locally trained model.
        mlflow.sklearn.log_model(
            sk_model=model,
            name="churn_pipeline",
            registered_model_name="customer_churn_pipeline",
            input_example=X_train.head(3),
            serialization_format="skops",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
            ],
        )

    return report


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))