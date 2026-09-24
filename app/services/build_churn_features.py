from pathlib import Path

import pandas as pd


PROCESSED_DATA_DIR = Path("data/processed")
RAW_DATA_DIR = Path("data/raw")

CUSTOMER_360_FILE = PROCESSED_DATA_DIR / "customer_360.csv"
POPULATION_FILE = RAW_DATA_DIR / "Telco_customer_churn_population.xlsx"

OUTPUT_FILE = PROCESSED_DATA_DIR / "churn_features.csv"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.strip("_")
    )

    return df


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    customer_360 = pd.read_csv(CUSTOMER_360_FILE)

    population = pd.read_excel(POPULATION_FILE)
    population = clean_columns(population)

    return customer_360, population


def enrich_with_population(
    customer_360: pd.DataFrame,
    population: pd.DataFrame,
) -> pd.DataFrame:

    print("\nPopulation columns:")
    print(population.columns.tolist())

    possible_zip_columns = [
        "zip_code",
        "zipcode",
        "zip",
    ]

    customer_zip_column = None
    population_zip_column = None

    for column in possible_zip_columns:
        if column in customer_360.columns:
            customer_zip_column = column

        if column in population.columns:
            population_zip_column = column

    if (
        customer_zip_column is None
        or population_zip_column is None
    ):
        print(
            "\nWARNING: Could not automatically identify "
            "ZIP code columns."
        )

        print(
            "Customer 360 ZIP candidates:",
            [
                column
                for column in customer_360.columns
                if "zip" in column
            ],
        )

        print(
            "Population ZIP candidates:",
            [
                column
                for column in population.columns
                if "zip" in column
            ],
        )

        return customer_360

    customer_360[customer_zip_column] = (
        customer_360[customer_zip_column]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.zfill(5)
    )

    population[population_zip_column] = (
        population[population_zip_column]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.zfill(5)
    )

    population = population.drop_duplicates(
        subset=[population_zip_column]
    )

    enriched = customer_360.merge(
        population,
        left_on=customer_zip_column,
        right_on=population_zip_column,
        how="left",
        suffixes=("", "_population"),
        validate="many_to_one",
    )

    return enriched


def create_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    features = df.copy()

    if "tenure_in_months" in features.columns:
        features["tenure_years"] = (
            features["tenure_in_months"] / 12
        ).round(2)

        features["is_new_customer"] = (
            features["tenure_in_months"] <= 6
        ).astype(int)

    if (
        "monthly_charge" in features.columns
        and "tenure_in_months" in features.columns
    ):
        features["estimated_lifetime_billing"] = (
            features["monthly_charge"]
            * features["tenure_in_months"]
        ).round(2)

    if "contract" in features.columns:
        features["is_month_to_month"] = (
            features["contract"]
            .astype(str)
            .str.lower()
            .eq("month-to-month")
            .astype(int)
        )

    if "customer_status" in features.columns:
        features["is_churned"] = (
            features["customer_status"]
            .astype(str)
            .str.lower()
            .eq("churned")
            .astype(int)
        )

    elif "churn_value" in features.columns:
        features["is_churned"] = (
            pd.to_numeric(
                features["churn_value"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )

    return features


def validate_features(
    df: pd.DataFrame,
) -> None:

    print("\n" + "=" * 80)
    print("CHURN FEATURE VALIDATION")
    print("=" * 80)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    if "customer_id" in df.columns:
        print(
            "Unique customers:",
            f"{df['customer_id'].nunique():,}",
        )

        print(
            "Duplicate customer IDs:",
            f"{df['customer_id'].duplicated().sum():,}",
        )

    if "is_churned" in df.columns:
        print("\nTarget distribution:")

        print(
            df["is_churned"]
            .value_counts(dropna=False)
            .sort_index()
            .to_string()
        )

        print(
            "\nChurn rate:",
            f"{df['is_churned'].mean() * 100:.2f}%",
        )

    print("\nEngineered features found:")

    engineered_columns = [
        "tenure_years",
        "is_new_customer",
        "estimated_lifetime_billing",
        "is_month_to_month",
        "is_churned",
    ]

    for column in engineered_columns:
        if column in df.columns:
            print(f" - {column}")


def main() -> None:
    print("InsightPilot - Build Churn Features")

    customer_360, population = load_data()

    enriched = enrich_with_population(
        customer_360,
        population,
    )

    features = create_features(
        enriched
    )

    validate_features(
        features
    )

    features.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\nChurn feature dataset saved to:"
        f" {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()