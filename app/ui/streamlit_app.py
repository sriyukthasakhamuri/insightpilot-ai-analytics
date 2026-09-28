import requests
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

API_URL = "http://127.0.0.1:8000"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="InsightPilot",
    page_icon="📊",
    layout="wide",
)


# =========================================================
# HEADER
# =========================================================

st.title(
    "📊 InsightPilot"
)

st.subheader(
    "Agentic AI Analytics Copilot"
)

st.write(
    (
        "Ask questions about customer churn risk, "
        "customer segments, retention policies, "
        "and customer-success actions."
    )
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "InsightPilot"
    )

    st.write(
        "AI-powered customer intelligence"
    )

    st.divider()

    st.write(
        "**Example questions**"
    )

    st.write(
        "• Who are the highest-risk customers?"
    )

    st.write(
        "• Which contract segment has the highest churn risk?"
    )

    st.write(
        "• What does our SLA say about Critical-risk customers?"
    )

    st.write(
        (
            "• What should we do for our highest-risk "
            "customers according to policy?"
        )
    )

    st.divider()

    st.caption(
        (
            "Powered by FastAPI, LangGraph, "
            "Ollama, ML analytics, and semantic RAG."
        )
    )


# =========================================================
# QUESTION INPUT
# =========================================================

question = st.text_area(
    "Ask InsightPilot a business question",
    placeholder=(
        "Example: What should we do for our "
        "highest-risk customers according to policy?"
    ),
    height=120,
)


ask_button = st.button(
    "Ask InsightPilot",
    type="primary",
)


# =========================================================
# API CALL
# =========================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "InsightPilot is analyzing..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question":
                            question.strip()
                    },
                    timeout=120,
                )

                if response.status_code == 200:

                    result = response.json()

                    st.success(
                        "Analysis complete"
                    )

                    st.subheader(
                        "InsightPilot Response"
                    )

                    st.write(
                        result["answer"]
                    )

                    sources = result.get(
                        "sources",
                        [],
                    )

                    if sources:

                        st.divider()

                        st.subheader(
                            "Retrieved Sources"
                        )

                        for source in sources:

                            st.write(
                                f"• {source}"
                            )

                    else:

                        st.caption(
                            (
                                "No policy documents were "
                                "required for this question."
                            )
                        )

                else:

                    st.error(
                        (
                            "InsightPilot API returned "
                            f"status code "
                            f"{response.status_code}."
                        )
                    )

                    st.write(
                        response.text
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    (
                        "Could not connect to the "
                        "InsightPilot API."
                    )
                )

                st.info(
                    (
                        "Make sure FastAPI is running "
                        "on http://127.0.0.1:8000."
                    )
                )

            except requests.exceptions.Timeout:

                st.error(
                    (
                        "The request took too long. "
                        "Please try again."
                    )
                )

            except Exception as exception:

                st.error(
                    "Unexpected error."
                )

                st.write(
                    str(exception)
                )