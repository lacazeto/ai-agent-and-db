import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Load DeepSeek-Coder
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

def read_file(file_path):
    """Read the content of a single file."""
    with open(file_path, "rb") as file:
        content = file.read()
    return content

# Function to read content of files in a directory
def read_files_in_directory(directory_path, file_extension=None):
    """Read all files in a directory and return their combined content."""
    combined_content = ""

    # Iterate over all files in the directory
    for root, dirs, files in os.walk(directory_path):
        for file_name in files:
            # Filter files by extension if provided
            if file_extension and not file_name.endswith(file_extension):
                continue

            file_path = os.path.join(root, file_name)
            with open(file_path, "rb") as file:
                content = file.read()
                combined_content += f"\n\n--- End of file: {file_name} ---\n\n{content}"

    return combined_content if combined_content else "No files found in the directory."

# Function to interact with the model
def ask_deepseek(question, context=""):
    """Use DeepSeek-Coder to answer questions based on a given context."""
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    print("Welcome to the DeepSeek-Coder terminal interface with folder reading!")
    print("Type 'exit' to quit.")
    
    while True:
        # Get user input
        action = input("\nEnter 'ask' to ask a question, 'read' to provide a folder path, or 'exit' to quit: ").strip().lower()

        combined_content = ""

        if action == 'exit':
            print("Exiting...")
            break

        if action == 'read':
            # Get the folder path from user
            folder_or_file_path = input("Enter the folder or filename path: ").strip()

            if os.path.isdir(folder_or_file_path):
                file_extension = input("Enter file extension to filter by (e.g., .txt, .py), or press Enter to include all files: ").strip() or None

                # Read files from the folder
                combined_content = read_files_in_directory(folder_or_file_path, file_extension)
                if combined_content.startswith("Error"):
                    print(combined_content)
                    continue

                print(f"\nFiles loaded successfully from the folder. You can now ask questions about the content.")
            else:
                combined_content = read_file(folder_or_file_path)
                print(f"\nFile loaded successfully. You can now ask questions about the content.")

            question = input("Enter your question: ")

            # Get the response from the model using the combined content
            response = ask_deepseek(question, combined_content)
            print(f"\nAnswer: {response}")

        elif action == 'ask':
            # Direct question without context (user provides only question and context manually)
            question = input("Enter your question: ")
            context = input("Enter context (optional, press Enter to skip): ")

            # Get the response from the model
            response = ask_deepseek(question, context)
            print(f"\nAnswer: {response}")
        
        else:
            print("Invalid action. Please type 'ask' or 'read'.")

if __name__ == "__main__":
    main()
