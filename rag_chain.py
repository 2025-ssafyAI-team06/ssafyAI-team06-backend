from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import ChatUpstage
from langchain_core.runnables import Runnable


def build_rag_chain() -> Runnable:
    template = (
        "당신은 제공된 정보를 바탕으로 질문에 답하는 도움이 되는 한국어 어시스턴트입니다.\n"
        "항상 주어진 컨텍스트만 활용하여 질문에 답해줘야 합니다.\n\n"
        "컨텍스트:\n{context}\n\n"
        "질문: {question}\n"
        "답변:"
    )

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatUpstage(model="solar-pro", temperature=0)
    rag_chain = prompt | llm | StrOutputParser()

    return rag_chain

rag_chain = build_rag_chain()

from langchain.schema import Document

def retrieve(state: dict, retriever) -> dict:
    print("---RETRIEVE---")
    question = state["question"]
    docs = retriever.get_relevant_documents(question)
    return {"documents": docs, "question": question}

def generate(state: dict, rag_chain) -> dict:
    print("---GENERATE ANSWER---")
    question = state["question"]
    docs = state["documents"]
    answer = rag_chain.invoke({"context": docs, "question": question})
    return {"question": question, "documents": docs, "generation": answer}