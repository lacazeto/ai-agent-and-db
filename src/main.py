import os
from utils.file_loader import load_code_files
from storage.chroma_store import ChromaStore
from utils.transformers import get_model_inputs, get_model_answer


def check_if_codebase_indexed(codebase_name):
    """Check if the codebase has been indexed."""
    if ChromaStore.collection_exists(codebase_name):
        print(f"Collection for {codebase_name} is already existent. Most probably has already been indexed. Would you like to (re)-index?")
        response = input("Enter 'Yes' to re-index or 'No' to skip: ")
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
        ChromaStore.store(filename, code)
    print("✅ Codebase indexed successfully!")

def search(codebase_name):
    """Runs the search query loop with batched embeddings."""
    ChromaStore.set_collection(codebase_name)

    while True:
        query = input("Enter your search query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        # Retrieve multiple relevant code snippets
        context = ChromaStore.query_db(query, top_k=3)

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
