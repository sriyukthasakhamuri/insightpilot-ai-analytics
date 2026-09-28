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

from app.rag.retrieval_service import (
    retrieve_documents,
)


# =========================================================
# AGENT STATE
# =========================================================


class AgentState(TypedDict):
    question: str
    answer: str
    sources: list[str]
    route: str


# =========================================================
# OLLAMA MODEL
# =========================================================


llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)


# =========================================================
# QUESTION ROUTER
# =========================================================


def route_question(
    state: AgentState,
) -> AgentState:

    question = state["question"].lower()

    # ---------------------------------
    # Policy indicators
    # ---------------------------------

    policy_keywords = [
        "policy",
        "sla",
        "playbook",
        "according to policy",
        "customer success",
        "retention action",
    ]

    # ---------------------------------
    # Segment analytics indicators
    # ---------------------------------

    segment_keywords = [
        "segment",
        "contract",
        "payment method",
        "internet type",
        "group",
        "compare",
        "highest churn rate",
    ]

    # ---------------------------------
    # Customer analytics indicators
    # ---------------------------------

    customer_keywords = [
        "highest-risk customers",
        "highest risk customers",
        "top risk",
        "top customers",
        "churn probability",
        "risk tier",
        "customer risk",
    ]

    has_policy = any(
        keyword in question
        for keyword in policy_keywords
    )

    has_segment = any(
        keyword in question
        for keyword in segment_keywords
    )

    has_customer = any(
        keyword in question
        for keyword in customer_keywords
    )

    # ---------------------------------
    # Route decision
    # ---------------------------------

    if has_policy and (
        has_customer or has_segment
    ):
        route = "combined"

    elif has_policy:
        route = "policy"

    elif has_segment:
        route = "segment"

    elif has_customer:
        route = "customer"

    else:
        route = "combined"

    state["route"] = route

    return state


# =========================================================
# ROUTE NODES
# =========================================================


def customer_route(
    state: AgentState,
) -> AgentState:

    state["route"] = "customer"

    return state


def segment_route(
    state: AgentState,
) -> AgentState:

    state["route"] = "segment"

    return state


def policy_route(
    state: AgentState,
) -> AgentState:

    state["route"] = "policy"

    return state


def combined_route(
    state: AgentState,
) -> AgentState:

    state["route"] = "combined"

    return state


# =========================================================
# CONDITIONAL ROUTE SELECTOR
# =========================================================


def choose_route(
    state: AgentState,
) -> str:

    return state["route"]


# =========================================================
# MAIN ANALYSIS NODE
# =========================================================


def analyze_question(
    state: AgentState,
) -> AgentState:

    question = state["question"]
    route = state["route"]

    context_sections = []
    source_names = []

    # =====================================================
    # 1. CUSTOMER RISK ANALYTICS
    # =====================================================

    if route in [
        "customer",
        "combined",
    ]:

        customers = get_top_risk_customers(
            limit=5
        )

        risk_summary = get_risk_summary()

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

        customer_section = f"""
OVERALL CUSTOMER RISK SUMMARY

Total customers:
{risk_summary['total_customers']}

Critical-risk customers:
{risk_summary['critical_customers']}

High-risk customers:
{risk_summary['high_customers']}

Medium-risk customers:
{risk_summary['medium_customers']}

Low-risk customers:
{risk_summary['low_customers']}

High or Critical customers:
{risk_summary['high_or_critical_customers']}

High or Critical percentage:
{risk_summary['high_or_critical_pct']}%

Average churn probability:
{risk_summary['average_churn_probability_pct']}%

TOP FIVE HIGHEST-RISK CUSTOMERS

{customer_context}
"""

        context_sections.append(
            customer_section
        )

    # =====================================================
    # 2. SEGMENT ANALYTICS
    # =====================================================

    if route in [
        "segment",
        "combined",
    ]:

        contract_analysis = analyze_segment(
            "contract"
        )

        payment_analysis = analyze_segment(
            "payment_method"
        )

        internet_analysis = analyze_segment(
            "internet_type"
        )

        segment_section = f"""
SEGMENT ANALYSIS

Contract analysis:
{contract_analysis}

Payment method analysis:
{payment_analysis}

Internet type analysis:
{internet_analysis}
"""

        context_sections.append(
            segment_section
        )

    # =====================================================
    # 3. POLICY / RAG RETRIEVAL
    # =====================================================

    if route in [
        "policy",
        "combined",
    ]:

        retrieved_documents = retrieve_documents(
            query=question,
            limit=3,
        )

        rag_sections = []

        for document in retrieved_documents:

            rag_sections.append(
                (
                    f"SOURCE: {document['source']}\n"
                    f"SIMILARITY: "
                    f"{document['similarity']}\n"
                    f"{document['content']}"
                )
            )

        rag_context = "\n\n".join(
            rag_sections
        )

        source_names = [
            document["source"]
            for document in retrieved_documents
        ]

        policy_section = f"""
RETRIEVED BUSINESS KNOWLEDGE

{rag_context}
"""

        context_sections.append(
            policy_section
        )

    # =====================================================
    # 4. COMBINE ONLY REQUIRED CONTEXT
    # =====================================================

    combined_context = "\n\n".join(
        context_sections
    )

    # =====================================================
    # 5. GROUNDED PROMPT
    # =====================================================

    prompt = f"""
You are InsightPilot, an AI analytics and
customer-retention assistant.

The user's question has been classified as:

ROUTE: {route}

Answer using only the information supplied
in the context below.

Do not invent:
- customer IDs
- percentages
- counts
- probabilities
- policies
- SLA requirements
- business facts

If the supplied information is insufficient,
say so clearly.

Do not describe model signals or associations
as proven causes of churn.

If the question asks about customer risk,
use the supplied customer analytics.

If the question asks about customer segments,
use the supplied segment analytics.

If the question asks about company policies,
SLA requirements, or retention procedures,
use only the retrieved business knowledge.

For retention prioritization, prioritize higher
churn risk or predicted churn rate unless the
supplied business policy says otherwise.

Keep the response concise and business-friendly.

AVAILABLE CONTEXT

{combined_context}

USER QUESTION

{question}
"""

    # =====================================================
    # 6. GENERATE RESPONSE
    # =====================================================

    response = llm.invoke(
        prompt
    )

    state["answer"] = response.content
    state["sources"] = source_names

    return state


# =========================================================
# BUILD LANGGRAPH
# =========================================================


def build_graph():

    graph = StateGraph(
        AgentState
    )

    # ---------------------------------
    # Nodes
    # ---------------------------------

    graph.add_node(
        "route_question",
        route_question,
    )

    graph.add_node(
        "customer_route",
        customer_route,
    )

    graph.add_node(
        "segment_route",
        segment_route,
    )

    graph.add_node(
        "policy_route",
        policy_route,
    )

    graph.add_node(
        "combined_route",
        combined_route,
    )

    graph.add_node(
        "analyze_question",
        analyze_question,
    )

    # ---------------------------------
    # Entry point
    # ---------------------------------

    graph.set_entry_point(
        "route_question"
    )

    # ---------------------------------
    # Conditional routing
    # ---------------------------------

    graph.add_conditional_edges(
        "route_question",
        choose_route,
        {
            "customer": "customer_route",
            "segment": "segment_route",
            "policy": "policy_route",
            "combined": "combined_route",
        },
    )

    # ---------------------------------
    # Route -> analysis
    # ---------------------------------

    graph.add_edge(
        "customer_route",
        "analyze_question",
    )

    graph.add_edge(
        "segment_route",
        "analyze_question",
    )

    graph.add_edge(
        "policy_route",
        "analyze_question",
    )

    graph.add_edge(
        "combined_route",
        "analyze_question",
    )

    graph.add_edge(
        "analyze_question",
        END,
    )

    return graph.compile()


# =========================================================
# LOCAL TEST
# =========================================================


def main():

    agent = build_graph()

    result = agent.invoke(
        {
           "question": (
              "What should we do for our "
              "highest-risk customers "
              "according to policy?"
            ),
            "answer": "",
            "sources": [],
            "route": "",
        }
    )

    print(
        "\nINSIGHTPILOT AI RESPONSE\n"
    )

    print(
        result["answer"]
    )

    print(
        "\nROUTE:"
    )

    print(
        result["route"]
    )

    print(
        "\nSOURCES:"
    )

    print(
        result["sources"]
    )


if __name__ == "__main__":
    main()