#pip install langchain langchain-core langchain-community langchain-upstage chromadb
#!pip install -q langchain-upstage chromadb unstructured
#!pip install -U langchain langchain-community langchain-core
import os
import getpass
import warnings

from dotenv import load_dotenv
load_dotenv()

os.environ["UPSTAGE_API_KEY"]

from langchain_community.document_loaders import WebBaseLoader
from langchain.schema import Document

from typing import List


def load_wikipedia_docs() -> List[Document]:
    # Years for which controversy pages exist
    years = [2006, 2010, 2014, 2018, 2022]
    wiki_urls = [
        f"https://en.wikipedia.org/wiki/List_of_{year}_FIFA_World_Cup_controversies"
        for year in years
    ]

    # Load documents
    wiki_loader = WebBaseLoader(wiki_urls)
    wiki_docs = wiki_loader.load()
    print(f"Retrieved {len(wiki_docs)} Wikipedia documents.")
    return wiki_docs

def load_namuwiki_docs(folder_path: str = "worldcup_incidents") -> List[Document]:
    namu_docs = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            full_path = os.path.join(folder_path, filename)
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            namu_docs.append(Document(
                page_content=content,
                metadata={
                    "source": filename,
                    "title": filename.replace(".txt", "")
                }
            ))

    print(f"총 {len(namu_docs)}개의 나무위키 문서가 수동으로 추가되었습니다.")
    return namu_docs



from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List

def build_retriever(wiki_docs: List[Document], namu_docs: List[Document], persist_dir: str = "./chroma_db"):
    # Combine all documents
    all_docs = wiki_docs + namu_docs

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(all_docs)
    print(f"✅ Total chunks created: {len(split_docs)}")

    # Initialize embedding model
    embedding_model = UpstageEmbeddings(model='solar-embedding-1-large')

    # Create Chroma vector store with embeddings
    vectorstore = Chroma.from_documents(split_docs, embedding_model, persist_directory=persist_dir)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    print("✅ Chroma vector store built with embeddings. Ready for queries.")

    return retriever

# retriever = build_retriever(wiki_docs, namu_docs)