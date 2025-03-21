import os

def load_code_files(directory: str):
    """Reads all code files in a directory and its subdirectories and returns their content."""
    code_snippets = {}
    for root, _, files in os.walk(directory):
        # Skip directories that are not relevant
        if any(excluded in root for excluded in ['node_modules', '.git', '.github']):
            continue
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, "rb") as f:
                code_snippets[file_path] = f.read()
            print(file_path)
    return code_snippets