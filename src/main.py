import os
from storage.chroma_store import HybridCodeIndexer
from utils.transformers import get_model_inputs, get_model_answer

def index_codebase(directory):
    last_entry = os.path.basename(directory)
    indexer = HybridCodeIndexer(last_entry)
    
    if indexer.collection_exists(last_entry):
        print(f"Collection for {last_entry} is already existent. Most probably has already been indexed. Would you like to (re)-index?")
        response = input("Enter 'Yes' to re-index or 'No' to skip: ")
        if response.lower() not in ['yes', 'y']:
            print("Skipping indexing process.")
            return

    print(f"Indexing codebase in directory: {last_entry}")
    
    """Indexes all code files in the specified directory."""
    indexer.index_codebase(directory)

def search(codebase_name):
    """Runs the search query loop with batched embeddings."""
    indexer = HybridCodeIndexer(codebase_name)

    while True:
        query = input("Enter your search query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        # Retrieve multiple relevant code snippets
        context = indexer.search_code(query, 5)

        # Create model input with larger context
        inputs = get_model_inputs(context, query)
       
        print(get_model_answer(inputs))

if __name__ == "__main__":
    print("🔹 Code Search CLI 🔹")
    print("1. Index Codebase")
    print("2. Ask about the Code")
    choice = input("Select an option: ")
    
    if choice == "1":
        directory = input("Enter the absolute path to your codebase: ")
        index_codebase(directory)
    elif choice == "2":
        print("🔍 Ask about the Code: Please make sure to go through step 1 before asking!")
        codebase_name = input("Enter the codebase name (ie. 'portal' for ~/.../portal): ")
        search(codebase_name)
