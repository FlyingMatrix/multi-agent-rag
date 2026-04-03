from typing import List

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, Node

from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents: List[Document], chunk_size: int=CHUNK_SIZE, chunk_overlap: int=CHUNK_OVERLAP) -> List[Node]:
    """
        Split documents into chunks (nodes)
    """
    if not documents:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    nodes = splitter.get_nodes_from_documents(documents)

    return nodes

