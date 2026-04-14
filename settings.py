import os
from dataclasses import dataclass, field    # @dataclass -> simplify class creation, field() -> customize how attributes behave
from typing import ClassVar, Dict


def get_env(key: str, default: str) -> str:
    return os.getenv(key, default)

def get_env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except ValueError:
        return default
    
@dataclass(frozen=True)
class Settings:
    # ---- Core ----
    llm_name: str = field(default_factory=lambda: get_env("LLM", "llama3"))
    embed_model_name: str = field(
        default_factory=lambda: get_env(
            "EMBED_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    vector_database_path: str = field(
        default_factory=lambda: get_env("VECTOR_DATABASE_PATH", "./vector_database")
    )

    # ---- Chunking ----
    chunk_size: int = field(default_factory=lambda: get_env_int("CHUNK_SIZE", 512))
    chunk_overlap: int = field(default_factory=lambda: get_env_int("CHUNK_OVERLAP", 50))

    # ---- Static model capabilities ----
    LLM_CONTEXT: ClassVar[Dict[str, Dict[str, int]]] = {
        "llama3": {"ctx": 8192, "reserve": 1000},
        "mistral": {"ctx": 8192, "reserve": 1000},
    }

    
