from pathlib import Path

import pandas as pd


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

DEMOGRAPHICS_FILE = RAW_DATA_DIR / "Telco_customer_churn_demographics.xlsx"
LOCATION_FILE = RAW_DATA_DIR / "Telco_customer_churn_location.xlsx"
POPULATION_FILE = RAW_DATA_DIR / "Telco_customer_churn_population.xlsx"
SERVICES_FILE = RAW_DATA_DIR / "Telco_customer_churn_services.xlsx"
STATUS_FILE = RAW_DATA_DIR / "Telco_customer_churn_status.xlsx"

OUTPUT_FILE = PROCESSED_DATA_DIR / "customer_360.csv"


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


def load_source_data() -> dict[str, pd.DataFrame]:
    tables = {
        "demographics": pd.read_excel(DEMOGRAPHICS_FILE),
        "location": pd.read_excel(LOCATION_FILE),
        "population": pd.read_excel(POPULATION_FILE),
        "services": pd.read_excel(SERVICES_FILE),
        "status": pd.read_excel(STATUS_FILE),
    }

    for name, df in tables.items():
        tables[name] = clean_columns(df)

    return tables


def validate_customer_key(
    name: str,
    df: pd.DataFrame,
) -> None:
    if "customer_id" not in df.columns:
        raise ValueError(
            f"{name} does not contain customer_id"
        )

    duplicate_count = df["customer_id"].duplicated().sum()

    print(
        f"{name}: "
        f"{len(df):,} rows | "
        f"{df['customer_id'].nunique():,} unique customers | "
        f"{duplicate_count:,} duplicate customer IDs"
    )


def build_customer_360(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:

    demographics = tables["demographics"]
    location = tables["location"]
    services = tables["services"]
    status = tables["status"]

    customer_360 = demographics.copy()

    customer_360 = customer_360.merge(
        location,
        on="customer_id",
        how="left",
        suffixes=("", "_location"),
        validate="one_to_one",
    )

    customer_360 = customer_360.merge(
        services,
        on="customer_id",
        how="left",
        suffixes=("", "_services"),
        validate="one_to_one",
    )

    customer_360 = customer_360.merge(
        status,
        on="customer_id",
        how="left",
        suffixes=("", "_status"),
        validate="one_to_one",
    )

    return customer_360


def validate_customer_360(
    customer_360: pd.DataFrame,
) -> None:

    print("\n" + "=" * 80)
    print("CUSTOMER 360 VALIDATION")
    print("=" * 80)

    print(f"Rows: {len(customer_360):,}")
    print(
        "Unique customers: "
        f"{customer_360['customer_id'].nunique():,}"
    )

    duplicate_customers = (
        customer_360["customer_id"]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate customer IDs: "
        f"{duplicate_customers:,}"
    )

    missing_customer_ids = (
        customer_360["customer_id"]
        .isna()
        .sum()
    )

    print(
        "Missing customer IDs: "
        f"{missing_customer_ids:,}"
    )

    print(
        "Columns: "
        f"{len(customer_360.columns):,}"
    )

    print("\nCustomer status distribution:")

    if "customer_status" in customer_360.columns:
        print(
            customer_360["customer_status"]
            .value_counts(dropna=False)
            .to_string()
        )

    print("\nChurn label distribution:")

    if "churn_label" in customer_360.columns:
        print(
            customer_360["churn_label"]
            .value_counts(dropna=False)
            .to_string()
        )


def main() -> None:
    print("InsightPilot - Build Customer 360")

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tables = load_source_data()

    print("\nSOURCE VALIDATION")

    for name in [
        "demographics",
        "location",
        "services",
        "status",
    ]:
        validate_customer_key(
            name,
            tables[name],
        )

    customer_360 = build_customer_360(
        tables
    )

    validate_customer_360(
        customer_360
    )

    customer_360.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\nCustomer 360 saved to:"
        f" {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()