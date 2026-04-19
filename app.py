import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# ---------------- UI ----------------
st.set_page_config(page_title="Local RAG Chat", layout="wide")
st.title("🤖 Chat with your Documents")

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["qwen2.5:7b", "llama3.2:1b"]
)

k_value = st.sidebar.slider("Chunks to retrieve (k)", 1, 6, 3)

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.chat_history = []

# ---------------- Load DB (FAST) ----------------
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    return db

try:
    db = load_db()
except:
    st.error("⚠️ Vector DB not found. Run ingest.py first.")
    st.stop()

retriever = db.as_retriever(search_kwargs={"k": k_value})

# ---------------- LLM ----------------
llm = OllamaLLM(
    model=model_choice,
    num_predict=400
)

# ---------------- Chat ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Ask something about your documents...")

if query:
    docs = retriever.invoke(query)

    context = "\n\n".join([
        f"[Source {i+1} | Page {doc.metadata.get('page','N/A')}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    ])

    prompt = f"""
You are an expert assistant.

Use the provided context to answer the question.

Guidelines:
- Give a detailed and well-structured answer
- Explain concepts clearly
- Use bullet points if helpful
- Add examples if possible
- Do NOT give one-line answers
- Cite sources like (Source 1, Page X)

If not found, say: "Not found in document".

Context:
{context}

Question:
{query}

Detailed Answer:
"""

    response = llm.invoke(prompt)

    st.session_state.chat_history.append(("user", query))
    st.session_state.chat_history.append(("ai", response))

    # Show sources
    with st.expander("📄 Sources Used"):
        for i, doc in enumerate(docs):
            st.markdown(f"**Source {i+1} | Page {doc.metadata.get('page','N/A')}**")
            st.write(doc.page_content[:500] + "...")

# ---------------- Chat UI ----------------
for role, message in st.session_state.chat_history:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(message)
