# RAG Mini

RAG has two phases:
1) Retrieval: chunk + embed + vector search (FAISS) to find relevant text.
2) Generation: an LLM writes an answer using the retrieved text as context.
