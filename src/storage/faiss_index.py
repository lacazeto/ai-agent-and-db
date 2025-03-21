import faiss
import numpy as np

dimension = 768  # Adjust based on the embedding model
index = faiss.IndexFlatL2(dimension)

def add_to_faiss(embedding: np.ndarray):
    """Add a vector to FAISS index."""
    index.add(np.array([embedding]))

def search_faiss(query_embedding: np.ndarray, k=3):
    """Search FAISS for the top-k most similar embeddings."""
    D, I = index.search(np.array([query_embedding]), k)
    return I, D  # Return indices and distances
