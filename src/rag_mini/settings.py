import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    embedding_model: str
    data_dir: str = "data"
    index_dir: str = "indexes"
    top_k: int = 4
    chunk_words: int = 200
    chunk_overlap_words: int = 50


def get_settings() -> Settings:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    return Settings(
        openai_api_key=key,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )


