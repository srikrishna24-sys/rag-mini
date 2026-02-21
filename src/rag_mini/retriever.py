from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from openai import OpenAI


@dataclass
class RetrievedChunk:
    score: float
    chunk_id: int
    doc_id: str
    source_path: str
    text: str


def load_metadata(meta_path: str) -> Dict[int, dict]:
    meta: Dict[int, dict] = {}
    with Path(meta_path).open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            meta[int(obj["chunk_id"])] = obj
    return meta


def embed_query(client: OpenAI, model: str, query: str) -> np.ndarray:
    resp = client.embeddings.create(model=model, input=[query.replace("\n", " ")])
    vec = np.array(resp.data[0].embedding, dtype=np.float32)[None, :]
    faiss.normalize_L2(vec)
    return vec


def retrieve(
    openai_api_key: str,
    embedding_model: str,
    index_dir: str,
    query: str,
    top_k: int,
) -> List[RetrievedChunk]:
    faiss_path = Path(index_dir) / "faiss.index"
    meta_path = Path(index_dir) / "metadata.jsonl"

    if not faiss_path.exists() or not meta_path.exists():
        raise RuntimeError("Index not found. Run: rag-mini ingest")

    index = faiss.read_index(str(faiss_path))
    meta = load_metadata(str(meta_path))

    client = OpenAI(api_key=openai_api_key)
    qvec = embed_query(client, embedding_model, query)

    scores, ids = index.search(qvec, top_k)

    out: List[RetrievedChunk] = []
    for score, cid in zip(scores[0].tolist(), ids[0].tolist()):
        if cid == -1:
            continue
        m = meta[int(cid)]
        out.append(
            RetrievedChunk(
                score=float(score),
                chunk_id=int(cid),
                doc_id=m["doc_id"],
                source_path=m["source_path"],
                text=m["text"],
            )
        )
    return out
