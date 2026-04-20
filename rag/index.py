from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from settings import Settings
from embedding import build_embed_model

import chromadb


settings = Settings()
EMBED_MODEL = build_embed_model(settings)
VECTOR_DATABASE_PATH = settings.vector_database_path


def build_index(nodes):
    """
        Build and persist a Chroma-backed index:
        takes text chunks (nodes) -> converts into embeddings -> stores in a vector database (Chroma)
    """
    chroma_client = chromadb.PersistentClient(path=VECTOR_DATABASE_PATH)        # start a persistent Chroma database
    collection = chroma_client.get_or_create_collection("rag_collection")       # create a collection for vectors
    vector_store = ChromaVectorStore(chroma_collection=collection)              # wrap with LlamaIndex adapter
    storage_context = StorageContext.from_defaults(vector_store=vector_store)   # create storage context -> tell LlamaIndex to use this vector store for storing data

    index = VectorStoreIndex(                                                   # build the index -> convert each node into a vector embedding using EMBED_MODEL
        nodes,
        storage_context=storage_context,
        embed_model=EMBED_MODEL
    )

    # return a searchable semantic index, now you can query it, build retrievals, and connect it to an LLM
    return index


def load_index():
    """
        Load existing index:
        reconnects to the stored data (without recomputing embeddings)
    """
    chroma_client = chromadb.PersistentClient(path=VECTOR_DATABASE_PATH)
    collection = chroma_client.get_or_create_collection("rag_collection")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )

    return index

