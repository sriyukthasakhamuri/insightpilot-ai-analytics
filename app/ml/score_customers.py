from pathlib import Path

import joblib
import pandas as pd


DATA_FILE = Path("data/processed/churn_features.csv")
MODEL_FILE = Path("models/churn_model.joblib")
OUTPUT_FILE = Path("data/processed/customer_churn_scores.csv")


LEAKAGE_COLUMNS = [
    "customer_id",
    "customer_status",
    "churn_label",
    "churn_value",
    "churn_score",
    "churn_category",
    "churn_reason",
    "cltv",
    "satisfaction_score",
    "status_id",
    "count_status",
    "quarter_status",
    "id",
]

TARGET_COLUMN = "is_churned"


def assign_risk_tier(probability: float) -> str:
    if probability >= 0.80:
        return "Critical"
    if probability >= 0.60:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


def main():
    print("InsightPilot - Score Customers")

    df = pd.read_csv(DATA_FILE)
    model = joblib.load(MODEL_FILE)

    customer_ids = df["customer_id"].copy()

    columns_to_remove = [
        column
        for column in LEAKAGE_COLUMNS
        if column in df.columns
    ]

    if TARGET_COLUMN in df.columns:
        columns_to_remove.append(TARGET_COLUMN)

    X = df.drop(
        columns=columns_to_remove,
        errors="ignore",
    )

    # Keep scoring features aligned with training
    expected_columns = model.feature_names_in_

    X = X.reindex(
        columns=expected_columns,
    )

    probabilities = model.predict_proba(X)[:, 1]

    predictions = model.predict(X)

    results = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "churn_probability": probabilities,
            "predicted_churn": predictions,
        }
    )

    results["churn_probability_pct"] = (
        results["churn_probability"] * 100
    ).round(2)

    results["risk_tier"] = results[
        "churn_probability"
    ].apply(assign_risk_tier)

    results = results.sort_values(
        "churn_probability",
        ascending=False,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nRisk tier distribution:")
    print(
        results["risk_tier"]
        .value_counts()
        .reindex(
            ["Critical", "High", "Medium", "Low"]
        )
        .fillna(0)
        .astype(int)
        .to_string()
    )

    print("\nTop 10 highest-risk customers:")
    print(
        results[
            [
                "customer_id",
                "churn_probability_pct",
                "risk_tier",
                "predicted_churn",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print(
        "\nCustomer risk scores saved to:"
        f" {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()