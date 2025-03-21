import os
from utils.file_loader import load_code_files
from utils.transformers import get_embedding, get_model_inputs, get_model_answer
from storage.chroma_store import ChromaStore
from search.query import search_embeddings

def check_if_codebase_indexed(codebase_name):
    """Check if the codebase has been indexed."""
    if ChromaStore.collection_has_data():
        print(f"Codebase {codebase_name} is already indexed. Would you like to re-index?")
        response = input("Enter 'yes' to re-index or 'no' to skip: ")
        if response.lower() in ['yes', 'y']:
            return False
        else:    
            print("Skipping indexing process.")
            return True
    return False

def index_codebase(directory):
    last_entry = os.path.basename(directory)
    ChromaStore.set_collection(last_entry)
    
    if check_if_codebase_indexed(last_entry):
        return

    print(f"Indexing codebase in directory: {last_entry}")
    
    """Indexes all code files in the specified directory."""
    code_files = load_code_files(directory)
    print(f"Found {len(code_files)} code files.")
    for filename, code in code_files.items():
        embedding = get_embedding(code).tolist()
        ChromaStore.store_embedding(filename, embedding, code)
    print("✅ Codebase indexed successfully!")

def search(codebase_name):
    """Runs the search query loop with batched embeddings."""
    ChromaStore.set_collection(codebase_name)

    while True:
        query = input("Enter your search query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        # Retrieve multiple relevant code snippets
        retrieved_snippets = search_embeddings(query, top_k=3)

        # Combine snippets to create a larger context
        context = "\n\n".join(retrieved_snippets)

        # Ensure it's within model token limit
        context = context[:4096]  # Truncate if needed

        # Create model input with larger context
        inputs = get_model_inputs(context, query)

        # Get answer
        return get_model_answer(inputs)

if __name__ == "__main__":
    print("🔹 Code Search CLI 🔹")
    print("1. Index Codebase")
    print("2. Search Code")
    choice = input("Select an option: ")
    
    if choice == "1":
        directory = input("Enter the path to your codebase: ")
        index_codebase(directory)
    elif choice == "2":
        print("🔍 Search Codebase: Please make sure to go through step 1 before searching!")
        codebase_name = input("Enter the codebase name: ")
        search(codebase_name)
