import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dataclasses import dataclass
from typing import ClassVar, Dict


@dataclass(frozen=True)
class Config:
    LLM_NAME: str = "llama3"                                            # llm
    LLM_CONTEXT: ClassVar[Dict[str, Dict[str, int]]] = {                # llm_context
        "llama3": {
            "ctx": 8192,
            "reserve": 1000
        },
    }
    EMBED_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"    # embeddings
    VECTOR_DATABASE_PATH: str = "./vector_database"                     # vector_database
    CHUNK_SIZE: int = 512                                               # Maximum size of each chunk (number of tokens per knowledge segment)
    CHUNK_OVERLAP: int = 50                                             # Number of overlapping tokens between consecutive chunks to preserve context

# ---- Builders ----

def get_llm_name(config: Config):
    return os.getenv("LLM", config.LLM_NAME)

def get_llm_context(config: Config):
    return config.LLM_CONTEXT[get_llm_name(config)]

def build_embed_model(config: Config):
    return HuggingFaceEmbedding(
        model_name=os.getenv("EMBED_MODEL", config.EMBED_MODEL_NAME)
    )

def get_vector_database_path(config: Config):
    return os.getenv("VECTOR_DATABASE_PATH", config.VECTOR_DATABASE_PATH)

def get_chunk_size(config: Config):
    return int(os.getenv("CHUNK_SIZE", config.CHUNK_SIZE))

def get_chunk_overlap(config: Config):
    return int(os.getenv("CHUNK_OVERLAP", config.CHUNK_OVERLAP))

# ---- Instantiate ----
config = Config()

LLM_NAME = get_llm_name(config)
LLM_CONTEXT = get_llm_context(config)
EMBED_MODEL = build_embed_model(config)
"""
    "all-MiniLM-L6-v2" is a SentenceTransformer model
    HuggingFaceEmbedding(...) wraps it into a proper embedding object
"""
VECTOR_DATABASE_PATH = get_vector_database_path(config)
"""
    Chroma creates a persistent on-disk vector database, composed of:
    - A SQLite metadata store
    - A vector index (HNSW) stored in binary files

    1. chroma.sqlite3 is the main metadata database, it stores:
    - Collections (e.g., "rag_collection")
    - Document IDs
    - Embedding references
    - Metadata (your doc.metadata, node metadata)
    - Index structure references

    2. The UUID folder (with an internal collection ID) is a Chroma collection storage directory
    - data_level0.bin stores the actual vector embeddings and contains high-dimensional vectors (most important)
    - header.bin stores metadata about the index structure and includes configuration like: vector dimensions and index parameters
    - length.bin stores lengths of stored vectors / segments and helps Chroma interpret the binary data correctly
    - link_lists.bin stores the HNSW (Hierarchical Navigable Small World) graph structure, which enables fast similarity search and approximate nearest neighbor lookup
"""
CHUNK_SIZE = get_chunk_size(config)          # Maximum size of each chunk (number of tokens per knowledge segment)
CHUNK_OVERLAP = get_chunk_overlap(config)    # Number of overlapping tokens between consecutive chunks to preserve context
"""
    typical chunking settings:
    - CHUNK_SIZE: 300–1000
    - CHUNK_OVERLAP: 10–20% of CHUNK_SIZE
"""

"""
    TODO: 
        - upgrade config into a typed, production-grade settings system
"""
