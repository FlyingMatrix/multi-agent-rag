from rag.index import load_index

class Retriever:
    """
        A Retriever class (system) to query relevant documents for RAG (Retrieval-Augmented Generation) 
        with LlamaIndex as the abstraction layer and using a vector database (ChromaDB)

        Here is what happens when user calls:

            retriever.retrieve("What is machine learning?")

        1. Query is embedded into a vector
        2. Compared against stored embeddings in Chroma
        3. Similarity search is performed
        4. Top k most similar chunks are returned
        5. These chunks are returned as response

        In addition: retrieve() vs query()
            - retrieve() -> returns raw nodes
            - query() -> may involve LLM response generation
    """
    def __init__(self, top_k: int = 5):
        self.index = load_index()                           # reconstructs a LlamaIndex VectorStoreIndex
        self.top_k = top_k                                  # controls how many results to return

        # converts the index into a query engine, which enables semantic search over embeddings
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=self.top_k,                    # returns the top_k most similar chunks
            response_mode="no_text"                         # prevents LLM from generating a response, only retrieves raw nodes (documents), making it as a pure retriever, not a generator
        )

    def retrieve(self, query: str):
        """
            Takes a user query (string), passes it to the query engine and returns retrieved results
        """
        response = self.query_engine.retrieve(query)
        return response
    
