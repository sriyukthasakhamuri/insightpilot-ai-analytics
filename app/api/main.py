from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.agents.risk_agent import build_graph


SCORES_FILE = Path(
    "data/processed/customer_churn_scores.csv"
)


app = FastAPI(
    title="InsightPilot API",
    version="2.0.0",
    description=(
        "AI analytics copilot for churn risk, "
        "customer segmentation, and retention intelligence."
    ),
)


# =========================================================
# AI AGENT
# =========================================================

insightpilot_agent = build_graph()


# =========================================================
# REQUEST MODEL
# =========================================================

class AskRequest(BaseModel):
    question: str


# =========================================================
# BASIC ENDPOINTS
# =========================================================

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


# =========================================================
# CUSTOMER RISK ENDPOINT
# =========================================================

@app.get("/customers/{customer_id}/risk")
def get_customer_risk(
    customer_id: str,
):

    if not SCORES_FILE.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "Customer churn scores "
                "file not found."
            ),
        )

    scores = pd.read_csv(
        SCORES_FILE
    )

    customer = scores[
        scores["customer_id"]
        == customer_id
    ]

    if customer.empty:
        raise HTTPException(
            status_code=404,
            detail="Customer not found.",
        )

    row = customer.iloc[0]

    return {
        "customer_id": row[
            "customer_id"
        ],
        "churn_probability_pct": float(
            row["churn_probability_pct"]
        ),
        "risk_tier": row[
            "risk_tier"
        ],
        "predicted_churn": int(
            row["predicted_churn"]
        ),
    }


# =========================================================
# AI ANALYTICS ENDPOINT
# =========================================================

@app.post("/ask")
def ask_insightpilot(
    request: AskRequest,
):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        result = insightpilot_agent.invoke(
            {
                "question": question,
                "answer": "",
            }
        )

        return {
            "question": question,
            "answer": result["answer"],
        }

    except Exception as exception:
        raise HTTPException(
            status_code=500,
            detail=(
                "InsightPilot could not "
                "process the question."
            ),
        ) from exception