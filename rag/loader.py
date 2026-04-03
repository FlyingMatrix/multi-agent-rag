from typing import List
from pathlib import Path

from llama_index.core.schema import Document
import pdfplumber

def extract_pdf_with_tables(file_path: Path) -> List[Document]:
    """
        Extract both text and tables from a PDF and store them in structured documents
    """
    docs = []

    with pdfplumber.open(file_path) as pdf:

        # iterates over all pages, page_num: page index (starting at 0), page: the actual page object
        for page_num, page in enumerate(pdf.pages): 
            text = page.extract_text() or ""

            # extracts tables from the page, return a list of tables, each table is a list of rows, each row is a list of cells
            tables = page.extract_tables()
            table_texts = []
            for table in tables:
                rows = [" | ".join(cell or "" for cell in row) for row in table]
                table_texts.append("\n".join(rows))

            combined_text = text + "\n\n" + "\n\n".join(table_texts)

            docs.append(
                Document(
                    text=combined_text,
                    metadata={
                        "source": str(file_path),
                        "page": page_num
                    }
                )
            )
    
    # return the full list of documents, each document represents one page of the PDF
    return docs

def load_documents(path: str) -> List[Document]:
    """
        Load documents (PDF, MD, TXT) and extracts text + basic tables, ignores images
    """
    path_obj = Path(path)

    if not path_obj.exists() or not path_obj.is_dir():
        raise ValueError(f"Invalid directory: {path}")

    documents = []

    for file_path in path_obj.rglob("*"):   
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            # read the PDF and return a list of Document objects (often one per page or chunk)
            docs = extract_pdf_with_tables(file_path)
            # add all returned Documents into the main documents list
            documents.extend(docs)

        elif suffix in [".md", ".txt"]:
            text = file_path.read_text(
                encoding="utf-8",   # encoding="utf-8" ensures proper decoding
                errors="ignore"     # errors="ignore" skips invalid characters instead of crashing
            )
            documents.append(
                Document(
                    text=text,
                    metadata={"source": str(file_path)}
                )
            )

    if not documents:
        print("Warning: No documents found.")

    return documents

