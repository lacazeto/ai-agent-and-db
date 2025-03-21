from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

# Load DeepSeek-Coder model
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def get_embedding(text: str) -> np.ndarray:
    """Generate an embedding using DeepSeek-Coder."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.mean(dim=1)
    return outputs.numpy().astype("float32").flatten()
