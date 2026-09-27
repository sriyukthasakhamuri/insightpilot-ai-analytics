from pathlib import Path

import numpy as np
from langchain_ollama import OllamaEmbeddings


KNOWLEDGE_BASE_DIR = Path(
    "docs/knowledge_base"
)


embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
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


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:

    a = np.array(vector_a)
    b = np.array(vector_b)

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


def retrieve_documents(
    query: str,
    limit: int = 3,
) -> list[dict]:

    documents = load_documents()

    query_embedding = (
        embedding_model.embed_query(
            query
        )
    )

    document_texts = [
        document["content"]
        for document in documents
    ]

    document_embeddings = (
        embedding_model.embed_documents(
            document_texts
        )
    )

    scored_documents = []

    for document, embedding in zip(
        documents,
        document_embeddings,
    ):

        similarity = cosine_similarity(
            query_embedding,
            embedding,
        )

        scored_documents.append(
            {
                "source": document["source"],
                "content": document["content"],
                "similarity": round(
                    similarity,
                    4,
                ),
            }
        )

    scored_documents.sort(
        key=lambda item: item[
            "similarity"
        ],
        reverse=True,
    )

    return scored_documents[:limit]


if __name__ == "__main__":

    query = (
        "What should the retention team do "
        "for a customer with very high churn risk?"
    )

    results = retrieve_documents(
        query=query,
        limit=3,
    )

    print(
        "\nSEMANTIC RAG RESULTS\n"
    )

    for result in results:

        print(
            f"Source: "
            f"{result['source']}"
        )

        print(
            f"Similarity: "
            f"{result['similarity']}"
        )

        print(
            result["content"][:700]
        )

        print(
            "\n" + "-" * 80 + "\n"
        )