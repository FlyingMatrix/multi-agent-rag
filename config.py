import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# ---- LLM ----
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# ---- Embeddings ----
"""
    "all-MiniLM-L6-v2" is a SentenceTransformer model
    HuggingFaceEmbedding(...) wraps it into a proper embedding object
"""
EMBED_MODEL = HuggingFaceEmbedding(
    model_name=os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
)

# ---- Vector DB ----
CHROMA_PATH = os.getenv("CHROMA_PATH", "./db")

# ---- Chunking ----
CHUNK_SIZE = 512        # Maximum size of each chunk (number of tokens per knowledge segment)
CHUNK_OVERLAP = 50      # Number of overlapping tokens between consecutive chunks to preserve context
"""
    typical chunking settings:
    - CHUNK_SIZE: 300–1000
    - CHUNK_OVERLAP: 10–20% of CHUNK_SIZE
"""

