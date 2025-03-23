import os

ignored_directories_and_files = ['node_modules', '.git', '.github', 'package-lock.json', 'yarn.lock', 'locales', 'dist', 'build', 'vendor', 'public', 'assets', 'images', 'fonts', 'css', 'js', 'static', 'coverage', 'docs', 'doc', 'example', 'examples']

def load_code_files(directory: str):
    """Reads all code files in a directory and its subdirectories and returns their content."""
    code_snippets = {}
    for root, _, files in os.walk(directory):
        # Skip directories that are not relevant
        if any(excluded in root for excluded in ignored_directories_and_files):
            continue
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                code_snippets[file_path] = f.read()
    return code_snippets