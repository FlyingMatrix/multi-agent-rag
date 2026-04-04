import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# ---- LLM ----
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# ---- Embeddings ----
EMBED_MODEL = HuggingFaceEmbedding(
    model_name=os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
)
"""
    "all-MiniLM-L6-v2" is a SentenceTransformer model
    HuggingFaceEmbedding(...) wraps it into a proper embedding object
"""

# ---- Vector Database ----
CHROMA_PATH = os.getenv("CHROMA_PATH", "./vector_database")
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

# ---- Chunking ----
CHUNK_SIZE = 512        # Maximum size of each chunk (number of tokens per knowledge segment)
CHUNK_OVERLAP = 50      # Number of overlapping tokens between consecutive chunks to preserve context
"""
    typical chunking settings:
    - CHUNK_SIZE: 300–1000
    - CHUNK_OVERLAP: 10–20% of CHUNK_SIZE
"""

