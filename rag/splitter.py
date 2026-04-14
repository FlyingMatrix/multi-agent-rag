from typing import List
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, Node
from settings import Settings

import re


settings = Settings()
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap

MAX_TABLE_CHARS = 3000  # threshold for large tables


def is_table(text: str, min_rows: int = 2, min_cols: int = 2) -> bool:
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # need enough rows
    if len(lines) < min_rows:
        return False

    split_patterns = [
        r"\t+",         # tabs
        r"\s{2,}",      # 2+ spaces
        r"\s*\|\s*",    # pipe with optional spaces
    ]

    best_match_counts = []

    for pattern in split_patterns:
        col_counts = []

        for line in lines:
            cols = re.split(pattern, line)
            cols = [c for c in cols if c.strip()]  # remove empty cells

            if len(cols) >= min_cols:
                col_counts.append(len(cols))

        # not enough structured rows
        if len(col_counts) < min_rows:
            continue

        best_match_counts.append(col_counts)

    if not best_match_counts:
        return False

    # evaluate consistency
    for counts in best_match_counts:
        unique_counts = set(counts)

        # strong signal: same number of columns across rows
        if len(unique_counts) == 1:
            return True

        # weaker signal: small variation allowed
        if len(unique_counts) <= 2:
            return True

    return False


def clean_nodes(nodes):
    return [
        node for node in nodes
        if node.get_content() and node.get_content().strip()
    ]


def split_documents(documents: List[Document], chunk_size: int=CHUNK_SIZE, chunk_overlap: int=CHUNK_OVERLAP) -> List[Node]:
    """
        Split documents into chunks (nodes) while preserving tables
        Strategies: 
            1. Detect table-like content
            2. Keep small tables as single chunks
            3. Only split normal text and large tables
    """
    if not documents:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    nodes = []

    for doc in documents:
        text = doc.text
        base_metadata = doc.metadata or {}

        # split into sections (basic heuristic)
        sections = re.split(r"\n{2,}", text)

        for section in sections:
            section = section.strip()
            if not section:
                continue

            if is_table(section):
                if len(section) > MAX_TABLE_CHARS:  # split large tables safely
                    rows = section.split("\n")
                    chunk = []
                    current_length = 0

                    for row in rows:
                        chunk.append(row)
                        current_length += len(row) + 1  # +1 for counting the newline "\n"

                        if current_length > MAX_TABLE_CHARS:
                            nodes.append(
                                Node(
                                    text="\n".join(chunk),
                                    metadata={**base_metadata, "type": "table"}
                                )
                            )
                            current_length = 0
                            chunk = []

                    if chunk:
                        nodes.append(
                            Node(
                                text="\n".join(chunk),
                                metadata={**base_metadata, "type": "table"}
                            )
                        )
                else:
                    # keep small table intact, preserved as ONE chunk
                    nodes.append(
                        Node(
                            text=section,
                            metadata={**base_metadata, "type": "table"} 
                        )
                    )
            else:
                # normal text -> split
                sub_nodes = splitter.get_nodes_from_documents(
                    [Document(text=section, metadata={**base_metadata, "type": "text"})]
                )
                nodes.extend(sub_nodes)

    return clean_nodes(nodes)

