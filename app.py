import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain.schema import Document
from pydantic import BaseModel
from embed import build_retriever

load_dotenv()

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 선언
class MessageRequest(BaseModel):
    message: str

# 전역 retriever 변수
retriever = None
chat_upstage = ChatUpstage(model="solar-pro")

@app.on_event("startup")
async def startup_event():
    global retriever
    retriever = build_retriever()  # 서버 시작 시 1회 실행
    print("🔁 Retriever initialized")

@app.post("/chat")
async def chat(req: MessageRequest):
    qa = RetrievalQA.from_chain_type(
        llm=chat_upstage,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    result = qa(req.message)
    return {
        "reply": result["result"],
        "sources": [doc.metadata.get("source", "") for doc in result["source_documents"]]
    }

@app.get("/")
async def root():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)