# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Cargamos un pipeline de question-answering con el modelo de mrm8488
qa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/bert-multi-cased-finetuned-xquadv1",
    tokenizer="mrm8488/bert-multi-cased-finetuned-xquadv1"
)

# Modelo Pydantic para la petición
class QAModel(BaseModel):
    question: str
    context: str

@app.get("/")
def read_root():
    return {"message": "La API funciona correctamente."}

@app.post("/answer")
def get_answer(data: QAModel):
    """
    Recibe question y context en español (u otro idioma soportado).
    Devuelve la mejor respuesta encontrada en el contexto.
    """
    result = qa_pipeline({
        "question": data.question,
        "context": data.context
    })
    # El pipeline retorna algo como:
    # { "score": 0.99, "start": 20, "end": 25, "answer": "París" }
    return {"answer": result["answer"], "score": result["score"]}
