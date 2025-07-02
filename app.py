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
import asyncio
import threading
from embed import build_retriever
from embed import load_namuwiki_docs, load_wikipedia_docs
from rag_chain import build_rag_chain, retrieve as rag_retrieve, generate as rag_generate  # ⬅ 추가


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
rag_chain = build_rag_chain()

@app.on_event("startup")
async def startup_event():
    # retriever 초기화는 별도 스레드에서 수행
    threading.Thread(target=init_retriever).start()

def init_retriever():
    global retriever
    try:
        wiki_docs = load_wikipedia_docs()
        namu_docs = load_namuwiki_docs("worldcup_incidents")
        retriever = build_retriever(wiki_docs, namu_docs)
        print("✅ Retriever initialized in background")
    except Exception as e:
        print("❌ Retriever initialization failed:", e)

@app.post("/chat")
async def chat(req: MessageRequest):
    global retriever
    if retriever is None:
        return {"reply": "❌ 아직 서버가 초기화되지 않았습니다. 잠시 후 다시 시도해주세요."}

    state = {"question": req.message}

    # rag_chain.py의 함수들로 처리
    state = rag_retrieve(state, retriever)
    state = rag_generate(state, rag_chain)

    return {
        "reply": state["generation"],
        "sources": [doc.metadata.get("source", "") for doc in state["documents"]]
    }

@app.get("/")
async def root():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)