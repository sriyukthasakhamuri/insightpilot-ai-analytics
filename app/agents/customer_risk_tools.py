from pathlib import Path

import pandas as pd


SCORES_FILE = Path(
    "data/processed/customer_churn_scores.csv"
)


def get_top_risk_customers(
    limit: int = 10,
) -> list[dict]:

    if not SCORES_FILE.exists():
        raise FileNotFoundError(
            "Customer churn scores file not found."
        )

    df = pd.read_csv(
        SCORES_FILE
    )

    columns = [
        "customer_id",
        "churn_probability_pct",
        "risk_tier",
        "predicted_churn",
    ]

    result = (
        df[columns]
        .sort_values(
            "churn_probability_pct",
            ascending=False,
        )
        .head(limit)
    )

    return result.to_dict(
        orient="records"
    )


if __name__ == "__main__":
    customers = get_top_risk_customers(
        limit=5
    )

    for customer in customers:
        print(customer)