import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from sentence_transformers import CrossEncoder

# ---------------- UI ----------------
st.set_page_config(page_title="Local RAG Chat", layout="wide")
st.title("🤖 Chat with your Documents")

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["qwen2.5:7b", "llama3.2:1b"]
)

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.history = []

# ---------------- Dynamic K ----------------
def get_k_values(query, model_name):
    q_len = len(query)

    if q_len < 30:
        retrieve_k = 6
    elif q_len < 100:
        retrieve_k = 8
    else:
        retrieve_k = 10

    if "1b" in model_name:
        final_k = 2
    else:
        final_k = 3

    return retrieve_k, final_k

# ---------------- Load DB ----------------
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

try:
    db = load_db()
except:
    st.error("⚠️ Vector DB not found. Run ingest.py first.")
    st.stop()

# ---------------- Reranker ----------------
@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

reranker = load_reranker()

# ---------------- LLM ----------------
llm = OllamaLLM(
    model=model_choice,
    num_predict=500,
    temperature=0.3
)

# ---------------- Memory ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- Chat Input ----------------
query = st.chat_input("Ask something about your documents...")

# 🚨 FIX: Only calculate k AFTER query exists
if query:

    retrieve_k, final_k = get_k_values(query, model_choice)

    retriever = db.as_retriever(search_kwargs={"k": retrieve_k})

    # Step 1: Better search query
    search_query = f"{query} detailed explanation with context"

    st.markdown(f"### 🔍 Search Query Used:\n`{search_query}`")

    # Step 2: Retrieve
    docs = retriever.invoke(search_query)

    # Step 3: Rerank
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    docs = [d for _, d in ranked[:final_k]]

    # Step 4: Context
    context = "\n\n".join([
        f"[Source {i+1} | File: {d.metadata.get('source_file','')} | Page/Row: {d.metadata.get('page', d.metadata.get('row','N/A'))}]\n{d.page_content}"
        for i, d in enumerate(docs)
    ])

    # Step 5: Memory
    history_text = "\n".join([
        f"User: {u}\nAI: {a}"
        for u, a in st.session_state.history[-3:]
    ])

    # Step 6: Prompt
    prompt = f"""
You are an expert assistant.

Conversation History:
{history_text}

Use the provided context to answer.

Guidelines:
- Give detailed answers
- Use bullet points
- Explain clearly
- Cite sources like (Source 1, File, Page/Row)
- If not found, say "Not found in document"

Context:
{context}

Question:
{query}

Answer:
"""

    # Step 7: LLM
    response = llm.invoke(prompt)

    # Step 8: Save
    st.session_state.chat_history.append(("user", query))
    st.session_state.chat_history.append(("ai", response))
    st.session_state.history.append((query, response))

    # ---------------- Sources UI ----------------
    with st.expander("📄 Sources Used"):
        for i, doc in enumerate(docs):
            st.markdown(
                f"**Source {i+1} | File: {doc.metadata.get('source_file','')} | Page/Row: {doc.metadata.get('page', doc.metadata.get('row','N/A'))}**"
            )

            text = doc.page_content
            if query.lower() in text.lower():
                text = text.replace(query, f"**{query}**")

            st.markdown(text[:700] + "...")

# ---------------- Chat UI ----------------
for role, message in st.session_state.chat_history:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(message)
