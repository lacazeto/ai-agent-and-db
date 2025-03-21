from embeddings.embedder import get_embedding
from storage.chroma_store import query_similar_code

def search_code(query: str):
    """Finds similar code snippets based on the user query."""
    query_embedding = get_embedding(query).tolist()
    results = query_similar_code(query_embedding, k=3)
    return results
