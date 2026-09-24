from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.agents.customer_risk_tools import (
    get_top_risk_customers,
)


class AgentState(TypedDict):
    question: str
    answer: str


def analyze_question(
    state: AgentState,
) -> AgentState:
    question = state["question"].lower()

    if (
        "highest-risk" in question
        or "highest risk" in question
        or "top risk" in question
        or "top churn" in question
    ):
        customers = get_top_risk_customers(
            limit=5
        )

        lines = [
            "Top 5 highest-risk customers:"
        ]

        for customer in customers:
            lines.append(
                f"- {customer['customer_id']}: "
                f"{customer['churn_probability_pct']}% "
                f"({customer['risk_tier']})"
            )

        state["answer"] = "\n".join(
            lines
        )

    else:
        state["answer"] = (
            "I can currently answer questions "
            "about highest-risk customers."
        )

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
    app = build_graph()

    result = app.invoke(
        {
            "question": (
                "Who are the highest-risk "
                "customers?"
            ),
            "answer": "",
        }
    )

    print(
        "\nINSIGHTPILOT RESPONSE\n"
    )

    print(
        result["answer"]
    )


if __name__ == "__main__":
    main()