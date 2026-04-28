# ingest.py

import os
import json
import pandas as pd

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------- CONFIG ----------------
DATA_PATH = "D:/LLAMA/docs"
DB_PATH = "./chroma_db"
TRACK_FILE = "processed_files.json"

# ---------------- TRACKING ----------------
def load_processed():
    if os.path.exists(TRACK_FILE):
        with open(TRACK_FILE, "r") as f:
            return json.load(f)
    return {}

def save_processed(data):
    with open(TRACK_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ---------------- EXCEL SMART LOADER ----------------
def load_excel_smart(file_path):
    df = pd.read_excel(file_path)

    docs = []
    for i, row in df.iterrows():
        content = "\n".join([f"{col}: {row[col]}" for col in df.columns])

        docs.append(Document(
            page_content=content,
            metadata={
                "source_file": os.path.basename(file_path),
                "row": int(i)
            }
        ))

    return docs

# ---------------- MAIN LOADER ----------------
def load_new_documents(folder_path, processed):
    docs = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if not os.path.isfile(file_path):
            continue

        last_modified = os.path.getmtime(file_path)

        # Skip unchanged files
        if file in processed and processed[file] == last_modified:
            print(f"⏩ Skipping unchanged: {file}")
            continue

        print(f"🆕 Processing: {file}")

        try:
            loaded_docs = []

            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                loaded_docs = loader.load()

            elif file.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                loaded_docs = loader.load()

            elif file.endswith(".xlsx") or file.endswith(".xls"):
                loaded_docs = load_excel_smart(file_path)

            elif file.endswith(".txt") or file.endswith(".md"):
                loader = TextLoader(file_path)
                loaded_docs = loader.load()

            else:
                print(f"⚠️ Skipping unsupported: {file}")
                continue

            # Add metadata
            for d in loaded_docs:
                d.metadata["source_file"] = file

            docs.extend(loaded_docs)

            # Update tracking
            processed[file] = last_modified

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

    return docs, processed

# ---------------- START ----------------
print("📂 Starting smart ingestion...")

processed = load_processed()

docs, processed = load_new_documents(DATA_PATH, processed)

if not docs:
    print("✅ No new or updated documents. Nothing to ingest.")
    exit()

print(f"✅ Loaded {len(docs)} new documents")

# ---------------- SPLIT ----------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = splitter.split_documents(docs)
print(f"✂️ Created {len(chunks)} chunks")

# ---------------- EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- VECTOR DB ----------------
print("💾 Updating vector DB...")

db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

db.add_documents(chunks)

# Save tracking
save_processed(processed)

print("🚀 Ingestion complete (incremental update)!")
