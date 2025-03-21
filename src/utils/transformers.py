from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

# To store vectors in FAISS (or any vector database like ChromaDB), we first need to transform text/code into vectors.

# Load DeepSeek-Coder model
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def get_embedding(text: str) -> np.ndarray:
    """Generate an embedding using DeepSeek-Coder."""
    print(f"Generating embedding for: {text}")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.mean(dim=1)
    return outputs.numpy().astype("float32").flatten()
    

def get_embeddings_batch(texts: list) -> np.ndarray:
    """Generate embeddings for a batch of texts using DeepSeek-Coder."""
    print(f"Generating embeddings for batch: {texts}")
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.mean(dim=1)
    return outputs.numpy().astype("float32")

def generate_inputs_from_embeddings(embeddings: np.ndarray) -> dict:
    """Generate inputs dictionary from embeddings for model inference."""
    return torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).to(model.device)

def get_model_inputs(context: str, query: str) -> dict:
    """Generate model inputs from context and query."""
    return tokenizer(f"Context: {context}\n\nQuestion: {query}", return_tensors="pt", truncation=True, padding=True).to(model.device)

def get_model_outputs(inputs: dict) -> np.ndarray:
    """Generate model outputs from inputs."""
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=4096)
    return outputs

def get_model_answer(inputs: dict) -> str:
    """Generate a model answer from inputs."""
    outputs = get_model_outputs(inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)