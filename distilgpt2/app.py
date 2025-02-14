from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

MODEL_NAME = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(request: PromptRequest):
    input_ids = tokenizer.encode(request.prompt, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=50)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"response": response}
