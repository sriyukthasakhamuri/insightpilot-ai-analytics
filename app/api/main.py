from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException


MODEL_FILE = Path("models/churn_model.joblib")
SCORES_FILE = Path("data/processed/customer_churn_scores.csv")


app = FastAPI(
    title="InsightPilot API",
    version="1.0.0",
    description="AI analytics API for customer churn intelligence.",
)


@app.get("/")
def root():
    return {
        "message": "InsightPilot API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.get("/customers/{customer_id}/risk")
def get_customer_risk(customer_id: str):
    if not SCORES_FILE.exists():
        raise HTTPException(
            status_code=500,
            detail="Customer churn scores file not found.",
        )

    scores = pd.read_csv(SCORES_FILE)

    customer = scores[
        scores["customer_id"] == customer_id
    ]

    if customer.empty:
        raise HTTPException(
            status_code=404,
            detail="Customer not found.",
        )

    row = customer.iloc[0]

    return {
        "customer_id": row["customer_id"],
        "churn_probability_pct": float(
            row["churn_probability_pct"]
        ),
        "risk_tier": row["risk_tier"],
        "predicted_churn": int(
            row["predicted_churn"]
        ),
    }