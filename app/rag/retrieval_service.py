from pathlib import Path
import re


KNOWLEDGE_BASE_DIR = Path(
    "docs/knowledge_base"
)


def load_documents() -> list[dict]:
    documents = []

    for file_path in KNOWLEDGE_BASE_DIR.glob("*.md"):
        content = file_path.read_text(
            encoding="utf-8"
        )

        documents.append(
            {
                "source": file_path.name,
                "content": content,
            }
        )

    return documents


def tokenize(text: str) -> set[str]:
    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower(),
    )

    stop_words = {
        "the",
        "and",
        "or",
        "to",
        "of",
        "a",
        "an",
        "is",
        "are",
        "for",
        "in",
        "on",
        "with",
        "be",
        "should",
    }

    return {
        word
        for word in words
        if word not in stop_words
    }


def score_document(
    query: str,
    document: str,
) -> int:

    query_tokens = tokenize(query)
    document_tokens = tokenize(document)

    return len(
        query_tokens.intersection(
            document_tokens
        )
    )


def retrieve_documents(
    query: str,
    limit: int = 3,
) -> list[dict]:

    documents = load_documents()

    scored_documents = []

    for document in documents:
        score = score_document(
            query,
            document["content"],
        )

        scored_documents.append(
            {
                "source": document["source"],
                "content": document["content"],
                "score": score,
            }
        )

    scored_documents.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return scored_documents[:limit]


if __name__ == "__main__":

    query = (
        "What should we do for "
        "Critical-risk customers?"
    )

    results = retrieve_documents(
        query=query,
        limit=3,
    )

    print(
        "\nRAG RETRIEVAL RESULTS\n"
    )

    for result in results:
        print(
            f"Source: {result['source']}"
        )

        print(
            f"Score: {result['score']}"
        )

        print(
            result["content"][:700]
        )

        print(
            "\n" + "-" * 80 + "\n"
        )