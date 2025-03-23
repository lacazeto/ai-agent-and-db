from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

DEVICE="cuda" if torch.cuda.is_available() else "cpu"

# Load DeepSeek-Coder model
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
embedding_model = AutoModel.from_pretrained(MODEL_NAME, device_map=DEVICE, trust_remote_code=True)
generation_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=DEVICE, trust_remote_code=True)


def get_embedding(text: str) -> np.ndarray:
    """Converts text into an embedding vector for similarity search (ChromaDB)."""
    print(f"Generating embedding for: {text}")

    # Format text consistently with model input
    formatted_text = f"### Context:\n{text}\n\n### Answer:\n"
    
    inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)

    with torch.no_grad():
        outputs = embedding_model(**inputs, output_hidden_states=True)

    # Extract last hidden state and pool it
    hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
    embedding = hidden_states.mean(dim=1)  # Mean Pooling
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)  # L2 normalize

    return embedding.cpu().numpy().astype("float32").flatten()

def get_model_inputs(context: str, query: str) -> dict:
    """Formats input into a structured prompt for text generation (LLM response)."""
    prompt = f"""You are a helpful AI assistant that analyzes code.
    
==== RETRIEVED CODE CONTEXT ====
{context}

==== USER QUERY ====
{query}

==== AI ANSWER ====
"""
    return tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(DEVICE)


def get_model_outputs(inputs: dict) -> torch.Tensor:
    """Generate model outputs from inputs."""
    with torch.no_grad():
        outputs = generation_model.generate(**inputs, max_new_tokens=512)
    return outputs

def get_model_answer(inputs: dict) -> str:
    """Generate a model answer from inputs."""
    outputs = get_model_outputs(inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
