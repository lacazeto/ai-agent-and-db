import os
from utils.file_loader import load_code_files
from embeddings.embedder import get_embedding
from storage.chroma_store import ChromaStore
from search.query import search_code

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

def search():
    """Runs the search query loop."""
    while True:
        query = input("Enter your search query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        results = search_code(query)
        print("🔍 Found matching code snippets:")
        for result in results:
            print(f"📂 {result['filename']}:\n{result['code']}\n{'-'*40}")

if __name__ == "__main__":
    print("🔹 Code Search CLI 🔹")
    print("1. Index Codebase")
    print("2. Search Code")
    choice = input("Select an option: ")
    
    if choice == "1":
        directory = input("Enter the path to your codebase: ")
        index_codebase(directory)
    elif choice == "2":
        search()
