# 📖 Mahabharata RAG - Local Production System

A Retrieval-Augmented Generation (RAG) system built using LangChain, FAISS, Ollama, and Streamlit.

This project allows users to ask questions about the Mahabharata and receive accurate answers retrieved directly from the book.

---

## 🚀 Features

- 🔎 Semantic search using FAISS
- 🧠 Local LLM (Llama3 via Ollama)
- 🔢 Local Embeddings (nomic-embed-text)
- 📚 Large PDF processing
- ⚡ Persistent FAISS index (no re-embedding on restart)
- 🖥 Streamlit interactive UI
- 💯 100% Local (No OpenAI API required)

---

## 🏗 Architecture

User Query  
↓  
FAISS Vector Search  
↓  
Retrieve Relevant Chunks  
↓  
Llama3 Generates Context-Based Answer  
↓  
Return Response in Streamlit UI  

---

## 📂 Project Structure
