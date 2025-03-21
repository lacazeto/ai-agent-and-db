import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel

# Load DeepSeek-Coder
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# FastAPI server
app = FastAPI()

class Query(BaseModel):
    question: str
    context: str = ""

@app.post("/ask")
def ask_deepseek(query: Query):
    """Use DeepSeek-Coder to answer questions."""
    prompt = f"Context:\n{query.context}\n\nQuestion:\n{query.question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)

    return {"response": tokenizer.decode(outputs[0], skip_special_tokens=True)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
