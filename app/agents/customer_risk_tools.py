from pathlib import Path

import pandas as pd


SCORES_FILE = Path(
    "data/processed/customer_churn_scores.csv"
)


def load_scores() -> pd.DataFrame:
    if not SCORES_FILE.exists():
        raise FileNotFoundError(
            "Customer churn scores file not found."
        )

    return pd.read_csv(SCORES_FILE)


def get_top_risk_customers(
    limit: int = 10,
) -> list[dict]:

    df = load_scores()

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


def get_risk_summary() -> dict:

    df = load_scores()

    total_customers = len(df)

    tier_counts = (
        df["risk_tier"]
        .value_counts()
        .to_dict()
    )

    critical_customers = int(
        tier_counts.get(
            "Critical",
            0,
        )
    )

    high_customers = int(
        tier_counts.get(
            "High",
            0,
        )
    )

    medium_customers = int(
        tier_counts.get(
            "Medium",
            0,
        )
    )

    low_customers = int(
        tier_counts.get(
            "Low",
            0,
        )
    )

    high_or_critical = (
        high_customers
        + critical_customers
    )

    high_or_critical_pct = (
        high_or_critical
        / total_customers
        * 100
    )

    average_risk = (
        df["churn_probability_pct"]
        .mean()
    )

    return {
        "total_customers": total_customers,
        "critical_customers": critical_customers,
        "high_customers": high_customers,
        "medium_customers": medium_customers,
        "low_customers": low_customers,
        "high_or_critical_customers": (
            high_or_critical
        ),
        "high_or_critical_pct": round(
            high_or_critical_pct,
            2,
        ),
        "average_churn_probability_pct": round(
            average_risk,
            2,
        ),
    }


if __name__ == "__main__":

    print(
        "\nTOP RISK CUSTOMERS\n"
    )

    customers = get_top_risk_customers(
        limit=5
    )

    for customer in customers:
        print(customer)

    print(
        "\nRISK SUMMARY\n"
    )

    summary = get_risk_summary()

    for key, value in summary.items():
        print(
            f"{key}: {value}"
        )