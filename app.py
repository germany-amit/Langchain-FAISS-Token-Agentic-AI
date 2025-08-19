import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ---- STREAMLIT UI ----
st.set_page_config(page_title="GenAI MVP", page_icon="🤖", layout="wide")
st.title("🧠 GenAI Architect Demo - RAG + Agentic AI + Guardrails (Open Source)")

uploaded_file = st.file_uploader("📄 Upload a text or PDF file", type=["txt", "pdf"])
query = st.text_input("💬 Ask your question:")

agent = st.radio("🤖 Choose Agent:", ["Agent 1 (Retriever)", "Agent 2 (Retriever + QA)"])

# ---- Load models ----
embedder = SentenceTransformer("all-MiniLM-L6-v2")   # free embeddings
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def load_pdf(file):
    if file.type == "application/pdf":
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
        return text
    else:
        return file.read().decode("utf-8")

if uploaded_file:
    text = load_pdf(uploaded_file)
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.split_text(text)

    # Create vector store
    doc_embeddings = embedder.encode(docs)
    vectorstore = FAISS.from_embeddings(doc_embeddings, docs, embedder)

    if query:
        # Agent 1: Simple retriever
        if agent == "Agent 1 (Retriever)":
            docs_with_scores = vectorstore.similarity_search(query, k=2)
            st.subheader("🤖 Agent 1 Answer:")
            for d in docs_with_scores:
                st.write(d.page_content)

        # Agent 2: Retriever + QA model
        elif agent == "Agent 2 (Retriever + QA)":
            docs_with_scores = vectorstore.similarity_search(query, k=2)
            context = " ".join([d.page_content for d in docs_with_scores])
            result = qa_model(question=query, context=context)
            st.subheader("🤖 Agent 2 Answer:")
            st.write(result["answer"])
