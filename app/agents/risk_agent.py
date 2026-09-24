from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from app.agents.customer_risk_tools import (
    get_top_risk_customers,
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

    customers = get_top_risk_customers(
        limit=5
    )

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

    prompt = f"""
You are InsightPilot, an AI analytics assistant.

Your job is to answer business questions using only the
customer risk data provided below.

Do not invent customer IDs, probabilities, or business facts.

Customer risk data:
{customer_context}

User question:
{question}

If the data is insufficient to answer the question,
say that clearly.

Provide a concise business-friendly answer.
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
                "Who are the highest-risk customers "
                "and what should the business focus on?"
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