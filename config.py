import os

# ---- LLM ----
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# ---- Embeddings ----
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# ---- Vector DB ----
CHROMA_PATH = os.getenv("CHROMA_PATH", "./rag_agent/db")

# ---- Chunking ----
CHUNK_SIZE = 512        # Maximum size of each chunk (number of tokens per knowledge segment)
CHUNK_OVERLAP = 50      # Number of overlapping tokens between consecutive chunks to preserve context
"""
    typical chunking settings:
    - CHUNK_SIZE: 300–1000
    - CHUNK_OVERLAP: 10–20% of CHUNK_SIZE
"""

