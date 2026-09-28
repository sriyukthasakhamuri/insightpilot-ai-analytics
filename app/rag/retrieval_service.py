from pathlib import Path
import re

import numpy as np
from langchain_ollama import OllamaEmbeddings


# =========================================================
# CONFIGURATION
# =========================================================

KNOWLEDGE_BASE_DIR = Path(
    "docs/knowledge_base"
)

EMBEDDING_MODEL = (
    "nomic-embed-text"
)

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


# =========================================================
# COSINE SIMILARITY
# =========================================================

def cosine_similarity(
    vector_a,
    vector_b,
) -> float:

    vector_a = np.array(
        vector_a,
        dtype=float,
    )

    vector_b = np.array(
        vector_b,
        dtype=float,
    )

    denominator = (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(
            vector_a,
            vector_b,
        )
        / denominator
    )


# =========================================================
# MARKDOWN SECTION PARSER
# =========================================================

def split_markdown_sections(
    content: str,
) -> list[dict]:

    lines = content.splitlines()

    sections = []

    current_heading = (
        "Document Overview"
    )

    current_lines = []

    for line in lines:

        stripped_line = line.strip()

        if re.match(
            r"^#{1,6}\s+",
            stripped_line,
        ):

            if current_lines:

                section_content = (
                    "\n".join(
                        current_lines
                    ).strip()
                )

                if section_content:

                    sections.append(
                        {
                            "section":
                                current_heading,
                            "content":
                                section_content,
                        }
                    )

            current_heading = re.sub(
                r"^#{1,6}\s+",
                "",
                stripped_line,
            ).strip()

            current_lines = []

        else:

            current_lines.append(
                line
            )

    if current_lines:

        section_content = (
            "\n".join(
                current_lines
            ).strip()
        )

        if section_content:

            sections.append(
                {
                    "section":
                        current_heading,
                    "content":
                        section_content,
                }
            )

    return sections


# =========================================================
# CHUNK LONG SECTIONS
# =========================================================

def chunk_section(
    text: str,
    max_words: int = 180,
    overlap_words: int = 30,
) -> list[str]:

    words = text.split()

    if len(words) <= max_words:

        return [
            text.strip()
        ]

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + max_words,
            len(words),
        )

        chunk = " ".join(
            words[start:end]
        ).strip()

        if chunk:

            chunks.append(
                chunk
            )

        if end >= len(words):
            break

        start = (
            end - overlap_words
        )

    return chunks


# =========================================================
# LOAD KNOWLEDGE BASE CHUNKS
# =========================================================

def load_knowledge_chunks() -> list[dict]:

    chunks = []

    markdown_files = sorted(
        KNOWLEDGE_BASE_DIR.glob(
            "*.md"
        )
    )

    for file_path in markdown_files:

        content = file_path.read_text(
            encoding="utf-8"
        )

        sections = (
            split_markdown_sections(
                content
            )
        )

        chunk_number = 0

        for section in sections:

            section_chunks = (
                chunk_section(
                    section["content"]
                )
            )

            for chunk_text in (
                section_chunks
            ):

                chunk_number += 1

                chunks.append(
                    {
                        "source":
                            file_path.name,
                        "section":
                            section[
                                "section"
                            ],
                        "chunk_index":
                            chunk_number,
                        "content":
                            chunk_text,
                    }
                )

    return chunks


# =========================================================
# SEMANTIC RETRIEVAL
# =========================================================

def retrieve_documents(
    query: str,
    limit: int = 3,
) -> list[dict]:

    chunks = (
        load_knowledge_chunks()
    )

    if not chunks:
        return []

    query_embedding = (
        embeddings.embed_query(
            query
        )
    )

    chunk_texts = [
        chunk["content"]
        for chunk in chunks
    ]

    chunk_embeddings = (
        embeddings.embed_documents(
            chunk_texts
        )
    )

    results = []

    for chunk, chunk_embedding in zip(
        chunks,
        chunk_embeddings,
    ):

        similarity = (
            cosine_similarity(
                query_embedding,
                chunk_embedding,
            )
        )

        results.append(
            {
                "source":
                    chunk["source"],
                "section":
                    chunk["section"],
                "chunk_index":
                    chunk[
                        "chunk_index"
                    ],
                "similarity":
                    round(
                        similarity,
                        4,
                    ),
                "content":
                    chunk["content"],
            }
        )

    results.sort(
        key=lambda item: (
            item["similarity"]
        ),
        reverse=True,
    )

    return results[:limit]


# =========================================================
# LOCAL TEST
# =========================================================

def main():

    query = (
        "What should the retention "
        "team do for a customer with "
        "very high churn risk?"
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
            f"Section: "
            f"{result['section']}"
        )

        print(
            f"Chunk: "
            f"{result['chunk_index']}"
        )

        print(
            f"Similarity: "
            f"{result['similarity']}"
        )

        print()

        print(
            result["content"]
        )

        print(
            "\n"
            + "-" * 60
            + "\n"
        )


if __name__ == "__main__":
    main()