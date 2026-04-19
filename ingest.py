# ingest.py

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DATA_PATH = "D:/LLAMA/docs"
DB_PATH = "./chroma_db"

print("📂 Loading PDFs...")

docs = []
for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_PATH, file))
        docs.extend(loader.load())

print(f"✅ Loaded {len(docs)} pages")

# Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = splitter.split_documents(docs)
print(f"✂️ Created {len(chunks)} chunks")

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create DB
print("💾 Creating vector DB...")

db = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    persist_directory=DB_PATH
)
print("🚀 Ingestion complete!")
