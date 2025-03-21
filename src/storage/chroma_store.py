import chromadb
from chromadb.api import Collection

chroma_client = chromadb.PersistentClient(path="./chroma_db")

def set_collection(collection_name: str):
    """Get or create a collection in ChromaDB."""
    global collection
    collection = chroma_client.get_or_create_collection(collection_name)
    
def store_embedding(file_name: str, embedding: list, code_snippet: str):
    """Store code embeddings in ChromaDB."""
    collection.add(
        ids=[file_name],
        embeddings=[embedding],
        metadatas=[{"filename": file_name, "code": code_snippet}]
    )

def query_similar_code(query_embedding: list, k=3):
    """Retrieve similar code snippets."""
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    return results["metadatas"]

def collection_has_data(self):
    """Check if the collection has any data."""
    if self.collection is None:
        raise ValueError("Collection is not set. Index a codebase first.")
    results = self.collection.query(query_embeddings=[[0]*len(self.collection.get_embedding_dimension())], n_results=1)
    return len(results["metadatas"]) > 0
