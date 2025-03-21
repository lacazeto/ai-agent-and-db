from utils.file_loader import load_code_files
from embeddings.embedder import get_embedding
from storage.chroma_store import store_embedding
from search.query import search_code

def index_codebase(directory):
    """Indexes all code files in the specified directory."""
    code_files = load_code_files(directory)
    for filename, code in code_files.items():
        embedding = get_embedding(code).tolist()
        store_embedding(filename, embedding, code)
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
