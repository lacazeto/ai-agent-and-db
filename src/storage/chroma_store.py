import chromadb

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("code_embeddings")

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
