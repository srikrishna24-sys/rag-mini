# rag-mini

Simple rag application for upskilling purpose

settings.py

Loads .env

Returns configuration like:

OPENAI_API_KEY

embedding model name

index folder name

chunk sizes

✅ Goal: keep all config in one place.

ingest.py (build the “library index”)

This runs when you do: rag-mini ingest

It:

Reads all .md/.txt from data/

Splits them into chunks (small pieces)

Calls OpenAI embeddings to convert each chunk into a vector

Stores vectors in FAISS

Saves metadata (chunk text + which file it came from)

✅ Output:

indexes/faiss.index (vectors)

indexes/metadata.jsonl (chunk text + source info)

retriever.py (find relevant chunks)

This runs during: rag-mini ask "..."

It:

Embeds your question into a vector

FAISS returns “closest chunk vectors”

Loads metadata to get actual chunk text + file path

✅ Output: top-k relevant chunks + similarity scores

generate.py (write the final answer)

This is the LLM step.

It:

Takes the retrieved chunks

Builds a “CONTEXT” block with citations

Calls OpenAI Responses API

LLM writes an answer only using that context

✅ Output: a clean answer + citations

cli.py (glue / command line app)

Defines commands:

rag-mini ingest

rag-mini ask "question"
