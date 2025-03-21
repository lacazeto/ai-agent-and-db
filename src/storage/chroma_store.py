import chromadb

class ChromaStore:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = None

    @staticmethod
    def set_collection(collection_name: str):
        """Get or create a collection in ChromaDB."""
        ChromaStore.collection = ChromaStore.chroma_client.get_or_create_collection(collection_name)
    
    @staticmethod
    def store_embedding(file_name: str, embedding: list, code_snippet: str):
        """Store code embeddings in ChromaDB."""
        if ChromaStore.collection is None:
            raise ValueError("Collection is not set. Call set_collection first.")
        ChromaStore.collection.add(
            ids=[file_name],
            embeddings=[embedding],
            metadatas=[{"filename": file_name, "code": code_snippet}]
        )

    @staticmethod
    def query_similar_code(query_embedding: list, k=3):
        """Retrieve similar code snippets."""
        if ChromaStore.collection is None:
            raise ValueError("Collection is not set. Call set_collection first.")
        results = ChromaStore.collection.query(query_embeddings=[query_embedding], n_results=k)
        return results["metadatas"]

    @staticmethod
    def collection_has_data():
        """Check if the collection has any data."""
        if ChromaStore.collection is None:
            raise ValueError("Collection is not set. Call set_collection first.")
        # Perform a query to check if the collection has any data
        try:
            results = ChromaStore.collection.query(query_embeddings=[[0.0]], n_results=1)
            return len(results["metadatas"]) > 0
        except Exception as e:
            # Handle the case where the collection is empty or the query fails
            return False
