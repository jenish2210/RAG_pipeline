import streamlit as st
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

st.set_page_config(page_title="Mahabharata RAG", layout="wide")

st.title("📖 Mahabharata RAG - Production")

INDEX_PATH = "faiss_index"

if not os.path.exists(INDEX_PATH):
    st.error("❌ FAISS index not found. Run build_index.py first.")
    st.stop()

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = FAISS.load_local(
    INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = Ollama(model="llama3")

query = st.text_input("Ask a question about Mahabharata:")

if query:
    with st.spinner("🔎 Retrieving relevant context..."):
        docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an expert on Mahabharata.
Answer ONLY from the context.
If not found, say 'Not found in document.'

Context:
{context}

Question:
{query}
"""

    with st.spinner("🧠 Generating answer..."):
        response = llm.invoke(prompt)

    st.subheader("📜 Answer")
    st.write(response)

    with st.expander("📚 Retrieved Chunks"):
        for i, doc in enumerate(docs):
            st.markdown(f"**Chunk {i+1}:**")
            st.write(doc.page_content)