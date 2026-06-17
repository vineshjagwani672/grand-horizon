from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path

try:
    from backend.rag import build_qa_chain
except ModuleNotFoundError:
    from rag import build_qa_chain

import os

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="Grand Horizon Hotel Knowledge Base Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

qa_chain = None


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.on_event("startup")
def startup_event():
    global qa_chain
    try:
        qa_chain = build_qa_chain()
    except Exception as exc:
        print(f"RAG startup warning: {exc}")
        qa_chain = None


@app.get("/")
def health_check():
    return {"message": "Grand Horizon Hotel Chatbot API is running"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    global qa_chain

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        if qa_chain is None:
            qa_chain = build_qa_chain()

        result = qa_chain.invoke({"query": question})
        return {"answer": result["result"]}

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {exc}") from exc
    
