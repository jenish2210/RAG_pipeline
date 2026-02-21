import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

INDEX_PATH = "faiss_index"

print("🚀 Starting Mahabharata Index Builder...\n")

# Step 1: Load PDF
print("📄 Loading PDF...")
loader = PyPDFLoader("mahabharata.pdf")
docs = loader.load()

print(f"✅ Total Pages in PDF: {len(docs)}")

# 🔥 Limit pages for first run (change later if needed)
LIMIT_PAGES = 400
docs = docs[:LIMIT_PAGES]

print(f"⚡ Using first {LIMIT_PAGES} pages for indexing\n")

# Step 2: Split Text
print("✂ Splitting text into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)

split_docs = splitter.split_documents(docs)

print(f"✅ Total Chunks Created: {len(split_docs)}\n")

# Step 3: Create Embeddings
print("🔢 Generating embeddings using Ollama (nomic-embed-text)...")
start_time = time.time()

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = FAISS.from_documents(split_docs, embeddings)

end_time = time.time()
print(f"⏱ Embedding Time: {round(end_time - start_time, 2)} seconds\n")

# Step 4: Save Index
print("💾 Saving FAISS index locally...")
vectorstore.save_local(INDEX_PATH)

print("\n✅ FAISS index created successfully!")
print("📁 Index folder created:", INDEX_PATH)
print("🎉 You can now run: streamlit run app.py")