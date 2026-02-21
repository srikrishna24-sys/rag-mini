from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import faiss
import numpy as np
from openai import OpenAI


@dataclass
class ChunkMeta:
    chunk_id: int
    doc_id: str
    source_path: str
    start_char: int
    end_char: int
    text: str


def iter_docs(data_dir: str) -> Iterable[Tuple[str, str, str]]:
    """Yield (doc_id, source_path, text) for .md/.txt files."""
    base = Path(data_dir)
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".md", ".txt"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
            doc_id = p.stem
            yield doc_id, str(p).replace("\\", "/"), text


def normalize_text(text: str) -> str:
    # Minimal normalization for stable chunking/embeddings
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")).strip()


def chunk_text_words(text: str, chunk_words: int, overlap_words: int) -> List[Tuple[int, int, str]]:
    """Simple word-based chunking (MVP). Returns (start_char, end_char, chunk_text)."""
    words = text.split()
    if not words:
        return []

    step = max(1, chunk_words - overlap_words)
    chunks: List[Tuple[int, int, str]] = []

    # Precompute prefix lengths for approximate char offsets
    prefix = [""] * (len(words) + 1)
    for i in range(1, len(words) + 1):
        prefix[i] = (prefix[i - 1] + (" " if i > 1 else "") + words[i - 1])

    for start in range(0, len(words), step):
        end = min(len(words), start + chunk_words)
        chunk = " ".join(words[start:end]).strip()
        start_char = len(prefix[start])
        end_char = start_char + len(chunk)

        if chunk:
            chunks.append((start_char, end_char, chunk))

        if end == len(words):
            break

    return chunks


def embed_texts(client: OpenAI, model: str, texts: List[str]) -> np.ndarray:
    """Return normalized embeddings as float32 numpy array."""
    resp = client.embeddings.create(model=model, input=[t.replace("\n", " ") for t in texts])
    vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
    faiss.normalize_L2(vectors)  # normalize so IP ~= cosine similarity
    return vectors


def build_index(
    openai_api_key: str,
    embedding_model: str,
    data_dir: str,
    index_dir: str,
    chunk_words: int,
    overlap_words: int,
) -> None:
    os.makedirs(index_dir, exist_ok=True)
    meta_path = Path(index_dir) / "metadata.jsonl"
    faiss_path = Path(index_dir) / "faiss.index"

    client = OpenAI(api_key=openai_api_key)

    metas: List[ChunkMeta] = []
    texts: List[str] = []
    chunk_id = 0

    for doc_id, source_path, raw in iter_docs(data_dir):
        text = normalize_text(raw)
        for start_char, end_char, chunk in chunk_text_words(text, chunk_words, overlap_words):
            metas.append(
                ChunkMeta(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    source_path=source_path,
                    start_char=start_char,
                    end_char=end_char,
                    text=chunk,
                )
            )
            texts.append(chunk)
            chunk_id += 1

    if not metas:
        raise RuntimeError(f"No .md/.txt files found under '{data_dir}'")

    vectors = embed_texts(client, embedding_model, texts)
    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, str(faiss_path))
    with meta_path.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")

    print(f"Indexed {len(metas)} chunks")
    print(f"FAISS: {faiss_path}")
    print(f"Meta : {meta_path}")
