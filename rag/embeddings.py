# rag/embedding.py

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()  # loads HUGGING_FACEHUB_API_TOKEN from .env

def get_embedding_model():
    """
    Returns a Hugging Face embedding model using the API token in .env
    """
    print("--- Connecting to Hugging Face Hub via API ---")
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"  # lightweight HF embedding model
        # API token is automatically read from HUGGING_FACEHUB_API_TOKEN
    )

def embed_documents(documents):
    """
    Takes a list of LangChain Document objects (chunks) and returns their embeddings
    """
    model = get_embedding_model()
    
    # Extract the text from each Document
    texts = [doc.page_content for doc in documents]
    
    # Generate embeddings
    embeddings = model.embed_documents(texts)
    
    print(f"Generated embeddings for {len(embeddings)} chunks.")
    return embeddings

# Optional: test run if executed directly
if __name__ == "__main__":
    from rag.loader import load_documents
    from rag.splitter import split_documents

    # 1. Load all documents from 'data' folder
    raw_docs = load_documents("data")
    
    # 2. Split them into chunks
    chunks = split_documents(raw_docs)
    
    # 3. Generate embeddings for all chunks
    embeddings = embed_documents(chunks)
    
    # 4. Show first 2 embeddings
    for i, emb in enumerate(embeddings[:2]):
        print(f"\n--- Embedding {i+1} (first 10 dimensions) ---")
        print(emb[:10])