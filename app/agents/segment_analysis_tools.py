from pathlib import Path

import pandas as pd


FEATURES_FILE = Path(
    "data/processed/churn_features.csv"
)

SCORES_FILE = Path(
    "data/processed/customer_churn_scores.csv"
)


def load_customer_data() -> pd.DataFrame:
    features = pd.read_csv(
        FEATURES_FILE
    )

    scores = pd.read_csv(
        SCORES_FILE
    )

    merged = features.merge(
        scores[
            [
                "customer_id",
                "churn_probability_pct",
                "risk_tier",
                "predicted_churn",
            ]
        ],
        on="customer_id",
        how="inner",
        validate="one_to_one",
    )

    return merged


def analyze_segment(
    segment_column: str,
) -> list[dict]:

    df = load_customer_data()

    if segment_column not in df.columns:
        raise ValueError(
            f"Segment column not found: "
            f"{segment_column}"
        )

    summary = (
        df.groupby(
            segment_column,
            dropna=False,
        )
        .agg(
            customers=(
                "customer_id",
                "count",
            ),
            average_churn_probability_pct=(
                "churn_probability_pct",
                "mean",
            ),
            predicted_churn_customers=(
                "predicted_churn",
                "sum",
            ),
        )
        .reset_index()
    )

    summary[
        "average_churn_probability_pct"
    ] = (
        summary[
            "average_churn_probability_pct"
        ]
        .round(2)
    )

    summary[
        "predicted_churn_rate_pct"
    ] = (
        summary[
            "predicted_churn_customers"
        ]
        / summary["customers"]
        * 100
    ).round(2)

    summary = summary.sort_values(
        "average_churn_probability_pct",
        ascending=False,
    )

    return summary.to_dict(
        orient="records"
    )


if __name__ == "__main__":

    for column in [
        "contract",
        "payment_method",
        "internet_type",
    ]:

        print(
            f"\nSEGMENT ANALYSIS: "
            f"{column}\n"
        )

        results = analyze_segment(
            column
        )

        for row in results:
            print(row)