# rag/vectorstore.py

import os
from langchain_chroma import Chroma
from rag.embeddings import get_embedding_model  # Your HF API logic
from tqdm import tqdm  # Optional, for progress bar if many chunks

def create_vector_db(chunks, db_path="./chroma_db"):
    """
    Creates a persistent Chroma vector database from document chunks using HF API embeddings.
    
    Args:
        chunks (List[Document]): List of LangChain Document chunks.
        db_path (str): Path to store Chroma database.
    
    Returns:
        Chroma: The vector database object.
    """
    if not chunks:
        print("No chunks provided! Cannot create vector DB.")
        return None

    # Step 1: Get the Hugging Face embedding model
    embeddings = get_embedding_model()
    
    # Step 2: Ensure the directory exists
    os.makedirs(db_path, exist_ok=True)
    
    # Step 3: Optional: show progress if many chunks
    print(f"--- Creating vector database with {len(chunks)} chunks ---")
    
    # Step 4: Create and persist the Chroma vector store
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )
    
    print(f"Vector database successfully saved at: {db_path}")
    return vector_db

# Optional test run
if __name__ == "__main__":
    from rag.loader import load_documents
    from rag.splitter import split_documents

    folder = "data"  # your folder containing text files
    raw_docs = load_documents(folder)
    if raw_docs:
        chunks = split_documents(raw_docs)
        create_vector_db(chunks)
    else:
        print("No documents found to index.")