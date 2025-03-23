import chromadb
from utils.transformers import get_embedding
import hashlib

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

        if ChromaStore.collection is None:
            raise COLLECTION_NOT_SET_ERROR

        embedding = get_embedding(code_snippet).tolist()
        unique_id = hashlib.md5(file_name.encode()).hexdigest()

        ChromaStore.collection.add(
            ids=[unique_id],
            embeddings=[embedding],
            metadatas=[{"filename": file_name, "code": code_snippet}]
        )

    @staticmethod
    def get_similar_code(query_embedding: list, k=3):
        """Retrieve similar code snippets with similarity scores."""
        if ChromaStore.collection is None:
            raise COLLECTION_NOT_SET_ERROR

        results = ChromaStore.collection.query(query_embeddings=[query_embedding], n_results=k)

        if not results or "metadatas" not in results:
            return []

        retrieved_snippets = [
            {
                "filename": meta["filename"],
                "code": meta["code"],
                "score": round(float(score), 4)  # Convert to float and round for readability
            }
            for meta, score in zip(results["metadatas"][0], results["distances"][0])  # Get distances
        ]

        # Sort by highest similarity (lowest distance)
        return sorted(retrieved_snippets, key=lambda x: x["score"])

    @staticmethod
    def collection_exists(collection_name: str) -> bool:
        """Check if the collection has any exists"""
        collections = ChromaStore.chroma_client.list_collections()
        return any(collection_name == str(collection) for collection in collections)

    @staticmethod
    def query_db(query: str, top_k: int = 3) -> list:
        """Search for the top-k most relevant code snippets."""
        query_embedding = get_embedding(query)  # Generate embedding for the query

        # Retrieve top-k similar vectors
        results = ChromaStore.get_similar_code(query_embedding, top_k)  # Returns (distances, indices)

        if not results:
            print("No relevant code snippets found.")
            return ""

         # Format results for the LLM
        retrieved_context = "\n\n".join(
            [f"File: {result['filename']} (Score: {result['score']})\nCode:\n{result['code']}" for result in results]
        )

        print(f"🔍 Retrieved {len(results)} code snippets.")

        return retrieved_context