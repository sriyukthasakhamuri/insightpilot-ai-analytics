from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from app.agents.customer_risk_tools import (
    get_risk_summary,
    get_top_risk_customers,
)

from app.agents.segment_analysis_tools import (
    analyze_segment,
)


class AgentState(TypedDict):
    question: str
    answer: str


llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)


def analyze_question(
    state: AgentState,
) -> AgentState:

    question = state["question"]

    # -----------------------------------------
    # 1. Overall customer risk information
    # -----------------------------------------

    customers = get_top_risk_customers(
        limit=5
    )

    risk_summary = get_risk_summary()

    # -----------------------------------------
    # 2. Segment-level analysis
    # -----------------------------------------

    contract_analysis = analyze_segment(
        "contract"
    )

    payment_analysis = analyze_segment(
        "payment_method"
    )

    internet_analysis = analyze_segment(
        "internet_type"
    )

    # -----------------------------------------
    # 3. Build customer context
    # -----------------------------------------

    customer_context = "\n".join(
        [
            (
                f"{customer['customer_id']} | "
                f"{customer['churn_probability_pct']}% | "
                f"{customer['risk_tier']}"
            )
            for customer in customers
        ]
    )

    # -----------------------------------------
    # 4. Build segment context
    # -----------------------------------------

    segment_context = f"""
Contract analysis:
{contract_analysis}

Payment method analysis:
{payment_analysis}

Internet type analysis:
{internet_analysis}
"""

    # -----------------------------------------
    # 5. Prompt the LLM
    # -----------------------------------------

    prompt = f"""
You are InsightPilot, an AI analytics assistant.

Answer the user's business question using only the
customer risk information provided below.

Do not invent customer IDs, percentages, counts,
probabilities, or business facts.

Overall risk summary:
Total customers: {risk_summary['total_customers']}
Critical-risk customers: {risk_summary['critical_customers']}
High-risk customers: {risk_summary['high_customers']}
Medium-risk customers: {risk_summary['medium_customers']}
Low-risk customers: {risk_summary['low_customers']}
High or Critical customers: {risk_summary['high_or_critical_customers']}
High or Critical percentage: {risk_summary['high_or_critical_pct']}%
Average churn probability: {risk_summary['average_churn_probability_pct']}%

Top five highest-risk customers:
{customer_context}

Segment analysis:
{segment_context}

User question:
{question}

When the user asks which segment should be prioritized for retention,
prioritize the segment with the highest churn risk or predicted churn rate,
not the segment with the lowest risk.

Base recommendations on the numerical evidence provided.

Clearly distinguish:
- highest-risk segment
- lower-risk comparison segments
- recommended business focus

Do not claim that a factor causes churn.
Describe relationships as associations or model signals.

If the available data is insufficient to answer the question,
say so clearly.

Use concise, business-friendly language.
"""

    response = llm.invoke(
        prompt
    )

    state["answer"] = response.content

    return state


def build_graph():

    graph = StateGraph(
        AgentState
    )

    graph.add_node(
        "analyze_question",
        analyze_question,
    )

    graph.set_entry_point(
        "analyze_question"
    )

    graph.add_edge(
        "analyze_question",
        END,
    )

    return graph.compile()


def main():

    agent = build_graph()

    result = agent.invoke(
        {
            "question": (
                "Which contract type should the "
                "retention team prioritize and why?"
            ),
            "answer": "",
        }
    )

    print(
        "\nINSIGHTPILOT AI RESPONSE\n"
    )

    print(
        result["answer"]
    )


if __name__ == "__main__":
    main()