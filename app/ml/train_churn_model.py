from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================================================
# PATHS
# =========================================================

DATA_FILE = Path("data/processed/churn_features.csv")

MODEL_DIR = Path("models")

MODEL_FILE = MODEL_DIR / "churn_model.joblib"

METRICS_FILE = MODEL_DIR / "churn_model_metrics.json"


# =========================================================
# COLUMNS THAT SHOULD NOT BE USED FOR TRAINING
# =========================================================

LEAKAGE_COLUMNS = [
    "customer_id",

    # Churn outcome / post-outcome information
    "customer_status",
    "churn_label",
    "churn_value",
    "churn_score",
    "churn_category",
    "churn_reason",
    "cltv",

    # Status-table fields that may leak outcome information
    "satisfaction_score",
    "status_id",
    "count_status",
    "quarter_status",

    # Non-predictive identifier
    "id",
]
TARGET_COLUMN = "is_churned"

# =========================================================
# LOAD DATA
# =========================================================

def load_training_data():

    df = pd.read_csv(DATA_FILE)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found."
        )

    print("=" * 80)
    print("INSIGHTPILOT - CHURN MODEL TRAINING")
    print("=" * 80)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    return df


# =========================================================
# PREPARE FEATURES
# =========================================================

def prepare_features(df):

    y = df[TARGET_COLUMN].astype(int)

    columns_to_remove = [
        column
        for column in LEAKAGE_COLUMNS
        if column in df.columns
    ]

    columns_to_remove.append(TARGET_COLUMN)

    X = df.drop(
        columns=columns_to_remove,
        errors="ignore",
    )

    # Remove extremely high-cardinality text / location fields
    high_cardinality_columns = []

    for column in X.select_dtypes(
        include=["object"]
    ).columns:

        unique_count = X[column].nunique(
            dropna=True
        )

        if unique_count > 100:
            high_cardinality_columns.append(
                column
            )

    if high_cardinality_columns:

        print(
            "\nDropping high-cardinality columns:"
        )

        for column in high_cardinality_columns:
            print(f" - {column}")

        X = X.drop(
            columns=high_cardinality_columns
        )

    numeric_columns = (
        X.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        X.select_dtypes(
            exclude=["number"]
        )
        .columns
        .tolist()
    )

    print(
        f"\nNumeric features: "
        f"{len(numeric_columns)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_columns)}"
    )

    return (
        X,
        y,
        numeric_columns,
        categorical_columns,
    )


# =========================================================
# BUILD PIPELINE
# =========================================================

def build_pipeline(
    numeric_columns,
    categorical_columns,
):

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ]
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


# =========================================================
# EVALUATION
# =========================================================

def evaluate_model(
    model,
    X_test,
    y_test,
):

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    metrics = {
        "accuracy": round(
            accuracy_score(
                y_test,
                predictions,
            ),
            4,
        ),

        "precision": round(
            precision_score(
                y_test,
                predictions,
            ),
            4,
        ),

        "recall": round(
            recall_score(
                y_test,
                predictions,
            ),
            4,
        ),

        "f1_score": round(
            f1_score(
                y_test,
                predictions,
            ),
            4,
        ),

        "roc_auc": round(
            roc_auc_score(
                y_test,
                probabilities,
            ),
            4,
        ),
    }

    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE")
    print("=" * 80)

    for metric, value in metrics.items():
        print(
            f"{metric:<12}: "
            f"{value:.4f}"
        )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
        )
    )

    return metrics


# =========================================================
# MAIN
# =========================================================

def main():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_training_data()

    (
        X,
        y,
        numeric_columns,
        categorical_columns,
    ) = prepare_features(
        df
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    print(
        f"\nTraining rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Testing rows: "
        f"{len(X_test):,}"
    )

    pipeline = build_pipeline(
        numeric_columns,
        categorical_columns,
    )

    print(
        "\nTraining Logistic Regression..."
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    metrics = evaluate_model(
        pipeline,
        X_test,
        y_test,
    )

    joblib.dump(
        pipeline,
        MODEL_FILE,
    )

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    print(
        "\nModel saved to:"
        f" {MODEL_FILE}"
    )

    print(
        "Metrics saved to:"
        f" {METRICS_FILE}"
    )

    print(
        "\nMODEL TRAINING COMPLETE"
    )


if __name__ == "__main__":
    main()