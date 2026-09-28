# InsightPilot

**Agentic AI Analytics Copilot for Customer Churn Intelligence**

InsightPilot is an end-to-end AI analytics application that combines machine learning, customer risk scoring, segment analysis, semantic RAG, LangGraph routing, FastAPI, Ollama, and Streamlit.

The project is designed to answer business questions such as:

- Who are the highest-risk customers?
- Which customer segment has the highest churn risk?
- What does our SLA say about Critical-risk customers?
- What should we do for our highest-risk customers according to policy?

---

## Overview

InsightPilot combines structured customer analytics with retrieval-augmented generation to provide grounded, business-focused answers.

Instead of sending every question through the same workflow, the LangGraph agent classifies each question and routes it to the appropriate analysis path.

Supported routes:

- **Customer Risk**
- **Segment Analysis**
- **Policy / RAG**
- **Combined Analytics + RAG**

This allows InsightPilot to use only the tools and context required for each question.

---

## Application Preview

### Home

![InsightPilot Home](./docs/01_home.png)

### Critical-Risk Policy Analysis

![Critical Risk Policy](./docs/02_critical_risk_policy.png)

### Segment Analysis

![Segment Analysis](./docs/03_segment_analysis.png)

---

## Key Features

### Customer 360 Analytics

Builds a consolidated customer-level dataset by combining demographics, location, services, account status, and population information.

The final Customer 360 dataset contains one record per customer and is used throughout the analytics and machine-learning pipeline.

---

### Churn Feature Engineering

Creates model-ready features such as:

- customer tenure
- contract characteristics
- payment behavior
- internet service information
- estimated lifetime billing
- new-customer indicators
- month-to-month contract indicators
- population enrichment

---

### Leakage-Controlled Churn Model

InsightPilot uses a logistic regression churn model with:

- median imputation for numerical features
- most-frequent imputation for categorical features
- numerical standardization
- one-hot encoding
- class balancing
- stratified train/test split

Potential target-leakage and post-outcome variables are excluded before training.

Model performance on the held-out test set:

- **Accuracy:** ~80.6%
- **Churn Recall:** ~86.6%
- **Churn Precision:** ~59.2%
- **Churn F1 Score:** ~70.4%

The relatively high recall is useful for retention scenarios where identifying customers at churn risk is important.

---

### Model Explainability

The logistic regression coefficients are extracted and transformed into human-readable churn signals.

The application distinguishes between:

- signals associated with higher modeled churn risk
- signals associated with lower modeled churn risk

These signals are treated as predictive associations rather than causal relationships.

---

### Customer Risk Scoring

Each customer receives:

- churn probability
- churn probability percentage
- predicted churn label
- risk tier

Risk tiers:

| Risk Tier | Churn Probability |
|---|---:|
| Critical | 80% or higher |
| High | 60%–79.99% |
| Medium | 40%–59.99% |
| Low | Below 40% |

Current scored customer population:

- **Total customers:** 7,043
- **Critical risk:** 1,346
- **High risk:** 989
- **Medium risk:** 799
- **Low risk:** 3,909
- **High or Critical:** 2,335
- **High or Critical share:** 33.15%
- **Average churn probability:** 38.29%

---

## Segment Analytics

InsightPilot can compare churn risk across business segments such as:

- contract type
- payment method
- internet type

Example result:

**Month-to-Month** customers were identified as the highest-risk contract segment in the current modeled dataset.

Observed metrics:

- average churn probability: ~62.67%
- predicted churn rate: ~70.22%
- predicted churn customers: 2,535

The agent uses these metrics to support retention prioritization questions.

---

## Agentic Question Routing

InsightPilot uses LangGraph to classify questions before analysis.

Example routes:

```text
"Who are the highest-risk customers?"
→ customer

"Which contract segment has the highest churn risk?"
→ segment

"What does our SLA say about Critical-risk customers?"
→ policy

"What should we do for our highest-risk customers according to policy?"
→ combined
```

The graph then routes the request to the appropriate analysis workflow.

This prevents unnecessary tools from running for every question.

---

## Semantic RAG

InsightPilot includes a business knowledge base containing:

- retention policy
- churn definitions
- customer success playbook
- support SLA

Knowledge-base documents are split by Markdown section and further chunked when required.

Each chunk includes metadata:

- source document
- section
- chunk index
- semantic similarity score

Embeddings are generated locally using:

```text
nomic-embed-text
```

through Ollama.

The retriever ranks chunks using cosine similarity and returns the most relevant business context to the agent.

---

## Grounded Policy Answers

For policy and combined questions, the LLM is instructed to:

- answer only from retrieved business knowledge
- preserve exact risk-tier terminology
- avoid inventing policies
- avoid inventing customer metrics
- distinguish Critical Risk from High Risk
- prioritize Critical-risk customers before lower-risk groups
- describe modeled relationships as associations rather than causes

Retrieved source metadata is returned separately through the API and displayed in the Streamlit interface.

Example source:

```text
retention_policy.md | Critical Risk | chunk 2
```

---

## System Architecture

```text
                     ┌─────────────────────────┐
                     │      Streamlit UI       │
                     │   Business Questions    │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │       FastAPI API       │
                     │       POST /ask         │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │     LangGraph Router    │
                     └────────────┬────────────┘
                                  │
             ┌────────────────────┼────────────────────┐
             │                    │                    │
             ▼                    ▼                    ▼
    ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
    │ Customer Risk  │   │ Segment        │   │ Policy / RAG   │
    │ Analytics      │   │ Analytics      │   │ Retrieval      │
    └────────┬───────┘   └────────┬───────┘   └────────┬───────┘
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │      Ollama LLM         │
                     │      llama3.2:3b        │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │ Grounded Business Answer│
                     │ + Retrieved Sources     │
                     └─────────────────────────┘
```

---

## Tech Stack

### AI / Agentic AI

- LangGraph
- LangChain Core
- Ollama
- llama3.2:3b
- nomic-embed-text
- semantic RAG

### Machine Learning

- Python
- pandas
- NumPy
- scikit-learn
- joblib
- logistic regression
- feature engineering
- explainability

### Backend

- FastAPI
- Uvicorn
- REST API

### Frontend

- Streamlit

### Data

- IBM Telco Customer Churn dataset
- Excel
- CSV
- Customer 360 analytics pipeline

### Version Control

- Git
- GitHub

---

## Project Structure

```text
insightpilot-ai-analytics/
│
├── app/
│   ├── agents/
│   │   ├── customer_risk_tools.py
│   │   ├── risk_agent.py
│   │   └── segment_analysis_tools.py
│   │
│   ├── api/
│   │   └── main.py
│   │
│   ├── ml/
│   │   ├── train_churn_model.py
│   │   ├── explain_churn_model.py
│   │   └── score_customers.py
│   │
│   ├── rag/
│   │   └── retrieval_service.py
│   │
│   ├── services/
│   │   ├── profile_telco_data.py
│   │   ├── build_customer_360.py
│   │   └── build_churn_features.py
│   │
│   └── ui/
│       ├── __init__.py
│       └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   ├── knowledge_base/
│   │   ├── churn_definitions.md
│   │   ├── customer_success_playbook.md
│   │   ├── retention_policy.md
│   │   └── support_sla.md
│   │
│   ├── 01_home.png
│   ├── 02_critical_risk_policy.png
│   └── 03_segment_analysis.png
│
├── models/
├── notebooks/
├── sql/
├── tests/
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

---

## API Endpoints

### Health Check

```http
GET /health
```

### Customer Risk

```http
GET /customers/{customer_id}/risk
```

Example response:

```json
{
  "customer_id": "8775-LHDJH",
  "churn_probability_pct": 98.98,
  "risk_tier": "Critical",
  "predicted_churn": 1
}
```

### Ask InsightPilot

```http
POST /ask
```

Example request:

```json
{
  "question": "What should we do for our highest-risk customers according to policy?"
}
```

Example response structure:

```json
{
  "question": "What should we do for our highest-risk customers according to policy?",
  "answer": "Grounded business response...",
  "sources": [
    "retention_policy.md | Critical Risk | chunk 2"
  ],
  "source_count": 1
}
```

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/sriyukthasakhamuri/insightpilot-ai-analytics.git
```

```bash
cd insightpilot-ai-analytics
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama models

```bash
ollama pull llama3.2:3b
```

```bash
ollama pull nomic-embed-text
```

Confirm:

```bash
ollama list
```

### 5. Start FastAPI

```bash
uvicorn app.api.main:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 6. Start Streamlit

Open a second terminal and activate the virtual environment:

```bash
source .venv/bin/activate
```

Then run:

```bash
streamlit run app/ui/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

---

## Example Questions

Try asking:

```text
Who are the highest-risk customers?
```

```text
Which contract segment has the highest churn risk?
```

```text
What does our SLA say about Critical-risk customers?
```

```text
What should we do for our highest-risk customers according to policy?
```

---

## Business Value

InsightPilot demonstrates how traditional analytics and modern AI can work together.

Instead of using an LLM as an isolated chatbot, the application combines:

- predictive churn modeling
- customer-level risk scoring
- segment-level analytics
- business policy retrieval
- question routing
- grounded natural-language generation

This allows business users to move from:

```text
"What happened?"
```

to:

```text
"Who is at risk?"
```

to:

```text
"Why does the model consider them risky?"
```

to:

```text
"What should the business do next?"
```

---

## Project Highlights

- Built an end-to-end Customer 360 analytics pipeline
- Developed a leakage-controlled churn prediction model
- Scored 7,000+ customers by churn probability and risk tier
- Added model explainability
- Built segment-level churn analytics
- Created FastAPI endpoints for customer intelligence
- Built a LangGraph-based routing agent
- Integrated local LLM inference with Ollama
- Implemented chunked semantic RAG using local embeddings
- Added policy-grounded answers with source metadata
- Built a Streamlit business-user interface
- Added route-specific execution to reduce unnecessary processing

---

## Future Improvements

Potential extensions include:

- persistent vector database such as Chroma or FAISS
- embedding caching
- automated RAG evaluation
- LangSmith or OpenTelemetry tracing
- conversation history
- customer lookup by ID from the UI
- interactive charts
- authentication
- Docker Compose
- cloud deployment
- automated tests
- CI/CD

---

## Author

Built as an AI Analytics Engineering portfolio project demonstrating:

**Machine Learning + Analytics Engineering + Agentic AI + RAG + APIs + Business Intelligence**

---

## License

This project is intended for educational and portfolio use.