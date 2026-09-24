from pathlib import Path

import joblib
import pandas as pd


MODEL_FILE = Path("models/churn_model.joblib")
OUTPUT_FILE = Path("data/processed/churn_feature_importance.csv")


def main():
    print("InsightPilot - Explain Churn Model")

    model_pipeline = joblib.load(MODEL_FILE)

    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    coefficients = model.coef_[0]

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    )

    importance["absolute_importance"] = (
        importance["coefficient"].abs()
    )

    importance["impact"] = importance[
        "coefficient"
    ].apply(
        lambda value: (
            "Higher Churn Risk"
            if value > 0
            else "Lower Churn Risk"
        )
    )

    importance = importance.sort_values(
        "absolute_importance",
        ascending=False,
    )

    importance.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nTop 20 churn drivers:")
    print(
        importance[
            [
                "feature",
                "coefficient",
                "impact",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print(
        "\nFeature importance saved to:"
        f" {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()