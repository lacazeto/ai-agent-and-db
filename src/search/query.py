from utils.transformers import get_embedding
from storage.chroma_store import ChromaStore

def search_embeddings(query: str, top_k: int = 3) -> list:
    """Search for the top-k most relevant code snippets."""
    query_embedding = get_embedding(query)  # Generate embedding for the query

    # Retrieve top-k similar vectors
    results = ChromaStore.query_similar_code(query_embedding, top_k)  # Returns (distances, indices)

   # Print the content of results to see what is coming
    print("Results:", results)

    # Extract the actual code snippets
    retrieved_snippets = [result['code'] for result in results]

    return retrieved_snippets  # Returns a list of code strings
