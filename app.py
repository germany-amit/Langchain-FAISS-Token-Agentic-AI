import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

# ---- STREAMLIT UI ----
st.set_page_config(page_title="GenAI MVP", page_icon="🤖", layout="wide")
st.title("🧠 GenAI Open-Source Demo - Two Agents on PDF")

uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])
query = st.text_input("💬 Ask your question:")

# ---- Load models ----
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")   # free embeddings
    qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    return embedder, qa_model

embedder, qa_model = load_models()

# ---- Extract text from PDF ----
def load_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

if uploaded_file:
    text = load_pdf(uploaded_file)

    # Split text into chunks
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)

    # Create embeddings + FAISS index
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    if query:
        # Encode query and retrieve top chunks
        q_emb = embedder.encode([query], convert_to_numpy=True)
        D, I = index.search(q_emb, k=2)
        retrieved_chunks = [chunks[i] for i in I[0]]

        col1, col2 = st.columns(2)

        # Agent 1: Retriever only
        with col1:
            st.subheader("🤖 Agent 1 (Retriever)")
            for c in retrieved_chunks:
                st.write(c)

        # Agent 2: Retriever + QA
        with col2:
            st.subheader("🤖 Agent 2 (Retriever + QA)")
            context = " ".join(retrieved_chunks)
            result = qa_model(question=query, context=context)
            st.write(result["answer"])
