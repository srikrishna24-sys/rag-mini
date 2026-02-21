from __future__ import annotations

from typing import List

from openai import OpenAI

from .retriever import RetrievedChunk


def build_context(chunks: List[RetrievedChunk]) -> str:
    """
    Build a context string that includes citations.
    Example citation format: [data/faq.md#chunk-3]
    """
    parts: List[str] = []
    for c in chunks:
        cite = f"[{c.source_path}#chunk-{c.chunk_id}]"
        parts.append(f"{cite}\n{c.text}")
    return "\n\n---\n\n".join(parts)


def generate_answer(
    openai_api_key: str,
    model: str,
    question: str,
    chunks: List[RetrievedChunk],
) -> str:
    """
    Use OpenAI Responses API to answer grounded in retrieved chunks.
    """
    client = OpenAI(api_key=openai_api_key)

    context = build_context(chunks)

    system = (
        "You are a helpful assistant.\n"
        "Answer using ONLY the provided CONTEXT.\n"
        "If the answer is not in the context, say exactly: Not found in the provided documents.\n"
        "Cite sources for key claims using the citation format included in the context.\n"
        "Do not invent citations."
    )

    user = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    # openai-python provides output_text convenience for Responses API
    return resp.output_text.strip()
