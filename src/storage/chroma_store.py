import os
import faiss
import numpy as np
import chromadb
from utils.transformers import get_embedding

IGNORED_DIRECTORIES_AND_FILES = {
    'node_modules', '.git', '.github', 'package-lock.json', 'yarn.lock', 'locales',
    'dist', 'build', 'vendor', 'public', 'assets', 'images', 'fonts', 'css', 'js',
    'static', 'coverage', 'docs', 'doc', 'example', 'examples'
}

class HybridCodeIndexer:
    def __init__(self, collection_name="codebase_index"):
        """Initializes FAISS and ChromaDB for hybrid storage and retrieval."""
        self.index = faiss.IndexFlatL2(2048)  # FAISS L2 search
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(collection_name)
        self.metadata = []  # Store metadata (file paths & content)

        # Load existing embeddings from ChromaDB into FAISS on startup
        self._load_from_chroma_to_faiss()

    def _load_from_chroma_to_faiss(self):
        """Load existing embeddings from ChromaDB into FAISS on startup."""
        all_docs = self.collection.get(include=['embeddings', 'metadatas'])
        if not all_docs["ids"]:
            print("ℹ️ No existing embeddings found in ChromaDB.")
            return

        embeddings = np.array(all_docs["embeddings"])
        if embeddings.size > 0:
            self.index.add(embeddings)
            self.metadata = all_docs["metadatas"]
            print(f"✅ Loaded {len(self.metadata)} embeddings from ChromaDB into FAISS.")

    def add_code(self, file_path, code):
        """Embeds the file content, stores it in ChromaDB, and indexes it in FAISS."""
        embedding = get_embedding(code)  # Generate embedding
        
        # Ensure FAISS accepts the format
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)  # Convert to 2D only if needed
        
        # Store in FAISS
        self.index.add(embedding)
        self.metadata.append({"file_path": file_path, "content": code})

        # Store in ChromaDB for persistence
        # Convert embedding to a list of lists for ChromaDB
        embedding_list = embedding.tolist()  # Convert numpy array to list of lists
        self.collection.add(
            ids=[file_path],  # Use file path as unique ID
            embeddings=embedding_list,  # Pass the embeddings correctly
            metadatas=[{"file_path": file_path, "content": code}]
        )

    def collection_exists(self, collection_name: str) -> bool:
        """Check if the collection has any exists"""
        collections = self.chroma_client.list_collections() 
        return any(collection_name == str(collection) for collection in collections)

    def search_code(self, query, top_k=3):
        """Searches FAISS index for similar code snippets."""
        query_embedding = get_embedding(query)

        # Ensure FAISS accepts the format
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)  # Convert to 2D only if needed

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):  # Ensure valid index
                results.append({
                    "file": self.metadata[idx]["file_path"],
                    "code": self.metadata[idx]["content"],
                    "score": round(float(distances[0][i]), 4)
                })

        return results

    def index_codebase(self, directory):
        """Reads all code files in a directory and indexes them using FAISS + ChromaDB."""
        for root, _, files in os.walk(directory):
            if any(ignored in root for ignored in IGNORED_DIRECTORIES_AND_FILES):
                continue

            for file in files:
                file_path = os.path.join(root, file)

                
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()
                    self.add_code(file_path, code)
            
        print(f"✅ Finished indexing codebase: {directory}")