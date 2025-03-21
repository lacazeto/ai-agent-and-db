import chromadb
from utils.transformers import get_embedding

COLLECTION_NOT_SET_ERROR = ValueError("Collection is not set. Call set_collection first.")

class ChromaStore:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = None

    @staticmethod
    def set_collection(collection_name: str):
        """Get or create a collection in ChromaDB."""
        ChromaStore.collection = ChromaStore.chroma_client.get_or_create_collection(collection_name)
    
    @staticmethod
    def store(file_name: str, code_snippet: str):
        """Store code embeddings in ChromaDB."""
        embedding = get_embedding(code_snippet).tolist()

        if ChromaStore.collection is None:
            raise COLLECTION_NOT_SET_ERROR

        ChromaStore.collection.add(
            ids=[file_name],
            embeddings=[embedding],
            metadatas=[{"filename": file_name, "code": code_snippet}]
        )

    @staticmethod
    def get_similar_code(query_embedding: list, k=3):
        """Retrieve similar code snippets."""
        if ChromaStore.collection is None:
            raise COLLECTION_NOT_SET_ERROR

        results = ChromaStore.collection.query(query_embeddings=[query_embedding], n_results=k)
        return results["metadatas"]

    @staticmethod
    def collection_exists(collection_name: str):
        """Check if the collection has any exists"""
        if ChromaStore.collection is None:
            raise COLLECTION_NOT_SET_ERROR
        
        collections = ChromaStore.chroma_client.list_collections()
        if collection_name in [str(collection) for collection in collections]:
            return True
        return False

    @staticmethod
    def query_db(query: str, top_k: int = 3) -> list:
        """Search for the top-k most relevant code snippets."""
        query_embedding = get_embedding(query)  # Generate embedding for the query

        # Retrieve top-k similar vectors
        results = ChromaStore.get_similar_code(query_embedding, top_k)  # Returns (distances, indices)

        # Extract the actual code snippets
        retrieved_snippets = [result['code'] for result in results[0]]

        print(f"Retrieved Snippets: {retrieved_snippets}")

        return "\n\n".join(retrieved_snippets)