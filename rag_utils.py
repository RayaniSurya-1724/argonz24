# rag_utils.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFLoader



GOOGLE_API_KEY = "AIzaSyAdii0tN49b5IF2XrYQ42nSn70nE4av8QA"  # 🔑 Replace with your API key

def clean_text(text: str) -> str:
    """Basic cleaning: remove line breaks, extra spaces, and artifacts."""
    text = text.replace("\n", " ")  # merge broken lines
    text = " ".join(text.split())   # normalize spaces
    return text

def build_vectorstore(pdf_path="docs/TEXTBOOK-CONSTRUCT-ON-STATISTICS-BAHRAIN-EN.pdf", index_path="faiss_index"):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # ✅ Clean each document
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    # ✅ Larger chunks for meaningful context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    # ✅ Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    # ✅ Store in FAISS
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_path)
    print(f"✅ FAISS index saved at {index_path}")

def load_vectorstore(index_path="faiss_index"):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

def query_rag(question, index_path="faiss_index", k=3):
    """Retrieve top-k answers from vectorstore"""
    db = load_vectorstore(index_path)
    docs = db.similarity_search(question, k=k)

    # ✅ Return clean text chunks
    return [doc.page_content for doc in docs]
