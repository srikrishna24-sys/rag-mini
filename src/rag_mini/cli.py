import typer

from .settings import get_settings
from .ingest import build_index
from .retriever import retrieve
from .generate import generate_answer

app = typer.Typer(help="rag-mini: tiny RAG app (OpenAI + FAISS)")


@app.command()
def ingest():
    """Build the FAISS index from files under ./data"""
    s = get_settings()
    build_index(
        openai_api_key=s.openai_api_key,
        embedding_model=s.embedding_model,
        data_dir=s.data_dir,
        index_dir=s.index_dir,
        chunk_words=s.chunk_words,
        overlap_words=s.chunk_overlap_words,
    )


@app.command()
def ask(
    q: str = typer.Argument(..., help="Your question"),
    top_k: int = typer.Option(None, help="How many chunks to retrieve (overrides default)"),
):
    """Ask a question using retrieval + OpenAI generation."""
    s = get_settings()
    k = top_k or s.top_k

    chunks = retrieve(
        openai_api_key=s.openai_api_key,
        embedding_model=s.embedding_model,
        index_dir=s.index_dir,
        query=q,
        top_k=k,
    )

    answer = generate_answer(
        openai_api_key=s.openai_api_key,
        model=s.openai_model,
        question=q,
        chunks=chunks,
    )

    typer.echo("\n=== ANSWER ===\n")
    typer.echo(answer)

    typer.echo("\n=== RETRIEVED CHUNKS ===\n")
    for c in chunks:
        typer.echo(f"- score={c.score:.4f}  [{c.source_path}#chunk-{c.chunk_id}]")
        snippet = c.text[:220].replace("\n", " ")
        typer.echo(f"  {snippet}")
        typer.echo("")


if __name__ == "__main__":
    app()
