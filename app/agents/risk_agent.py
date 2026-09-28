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


class AgentState(TypedDict):
    question: str
    answer: str
    sources: list[str]
    route: str

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)
def route_question(
    state: AgentState,
) -> AgentState:

    question = state["question"].lower()

    policy_keywords = [
        "policy",
        "sla",
        "playbook",
        "retention action",
        "what should we do",
        "according to policy",
        "customer success",
    ]

    segment_keywords = [
        "segment",
        "contract",
        "payment method",
        "internet type",
        "group",
        "compare",
        "highest churn rate",
    ]

    customer_keywords = [
        "customer",
        "customers",
        "highest risk",
        "top risk",
        "critical risk",
        "churn probability",
        "risk tier",
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

    if has_policy and (
        has_segment or has_customer
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

def choose_route(
    state: AgentState,
) -> str:

    return state["route"]


def analyze_question(
    state: AgentState,
) -> AgentState:

    question = state["question"]

    # =====================================================
    # 1. CUSTOMER RISK ANALYTICS
    # =====================================================

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

    # =====================================================
    # 2. SEGMENT ANALYTICS
    # =====================================================

    contract_analysis = analyze_segment(
        "contract"
    )

    payment_analysis = analyze_segment(
        "payment_method"
    )

    internet_analysis = analyze_segment(
        "internet_type"
    )

    segment_context = f"""
Contract analysis:
{contract_analysis}

Payment method analysis:
{payment_analysis}

Internet type analysis:
{internet_analysis}
"""

    # =====================================================
    # 3. RAG DOCUMENT RETRIEVAL
    # =====================================================

    retrieved_documents = retrieve_documents(
        query=question,
        limit=3,
    )

    rag_sections = []

    for document in retrieved_documents:

        rag_sections.append(
            (
                f"SOURCE: {document['source']}\n"
                f"SIMILARITY: {document['similarity']}\n"
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

    # =====================================================
    # 4. BUILD GROUNDED PROMPT
    # =====================================================

    prompt = f"""
You are InsightPilot, an AI analytics and customer-retention assistant.

Answer the user's question using only the information supplied below.

You have three information sources:

1. Customer churn analytics
2. Customer segment analytics
3. Retrieved business-policy documents

Do not invent:
- customer IDs
- percentages
- counts
- probabilities
- company policies
- SLA requirements
- business facts

If the information provided is insufficient, say so clearly.

Do not describe an association as proven causation.

When the user asks which customer segment should be prioritized for
retention, prioritize the segment with the highest churn risk or
predicted churn rate unless the business-policy context states otherwise.

When answering a policy question, base the answer primarily on the
retrieved policy documents.

--------------------------------------------------
OVERALL CUSTOMER RISK SUMMARY
--------------------------------------------------

Total customers: {risk_summary['total_customers']}
Critical-risk customers: {risk_summary['critical_customers']}
High-risk customers: {risk_summary['high_customers']}
Medium-risk customers: {risk_summary['medium_customers']}
Low-risk customers: {risk_summary['low_customers']}

High or Critical customers:
{risk_summary['high_or_critical_customers']}

High or Critical percentage:
{risk_summary['high_or_critical_pct']}%

Average churn probability:
{risk_summary['average_churn_probability_pct']}%

--------------------------------------------------
TOP FIVE HIGHEST-RISK CUSTOMERS
--------------------------------------------------

{customer_context}

--------------------------------------------------
SEGMENT ANALYSIS
--------------------------------------------------

{segment_context}

--------------------------------------------------
RETRIEVED BUSINESS KNOWLEDGE
--------------------------------------------------

{rag_context}

--------------------------------------------------
USER QUESTION
--------------------------------------------------

{question}

Provide a concise, business-friendly answer.

When business-policy documents are used, finish with:

Sources: <document names>

Only list sources that were actually retrieved.

Retrieved source names:
{source_names}
"""

    # =====================================================
    # 5. LLM RESPONSE
    # =====================================================

    response = llm.invoke(
        prompt
    )

    state["answer"] = response.content
    state["sources"] = source_names

    return state


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

def main():

    agent = build_graph()

    result = agent.invoke(
        {
            "question": (
                "What should we do for a "
                "Critical-risk customer "
                "according to policy?"
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
    print(
    "\nROUTE:"
)

    print(
        result["route"]
)


if __name__ == "__main__":
    main()