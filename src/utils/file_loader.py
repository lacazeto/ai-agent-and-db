import os

def load_code_files(directory: str):
    """Reads all code files in a directory and returns their content."""
    code_snippets = {}
    for file in os.listdir(directory):
        if file.endswith(".py"):  # Adjust for different languages
            with open(os.path.join(directory, file), "r", encoding="utf-8") as f:
                code_snippets[file] = f.read()
    return code_snippets
