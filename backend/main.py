from fastapi import FastAPI, Request
from pydantic import BaseModel
from rag_engine import get_answer

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_rag(query: Query):
    response = get_answer(query.question)
    return {"answer": response}
