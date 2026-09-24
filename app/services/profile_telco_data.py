from pathlib import Path

import pandas as pd


RAW_DATA_DIR = Path("data/raw")

FILES = {
    "demographics": "Telco_customer_churn_demographics.xlsx",
    "location": "Telco_customer_churn_location.xlsx",
    "population": "Telco_customer_churn_population.xlsx",
    "services": "Telco_customer_churn_services.xlsx",
    "status": "Telco_customer_churn_status.xlsx",
}


def profile_dataframe(name: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print(f"TABLE: {name}")
    print("=" * 80)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumn names:")
    for column in df.columns:
        print(f" - {column}")

    print("\nMissing values:")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("No missing values")
    else:
        print(missing.to_string())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))


def main() -> None:
    print("InsightPilot - Telco Data Profiling")

    for name, filename in FILES.items():
        file_path = RAW_DATA_DIR / filename

        if not file_path.exists():
            print(f"\nERROR: File not found: {file_path}")
            continue

        df = pd.read_excel(file_path)

        profile_dataframe(
            name=name,
            df=df,
        )


if __name__ == "__main__":
    main()