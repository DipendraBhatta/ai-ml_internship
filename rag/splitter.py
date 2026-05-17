# rag/splitter.py

from langchain_text_splitters import RecursiveCharacterTextSplitter

# IMPORT your loader here so the test works
from rag.loader import load_documents 

def split_documents(docs):
    # We keep it simple: 1000 chars per chunk, 100 char overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=100
    )
    return splitter.split_documents(docs)

if __name__ == "__main__":
    # 1. Load the docs using your loader.py logic
    # Make sure 'data' folder exists and has .txt files!
    raw_docs = load_documents("data") 
    
    
        # Step 2: Split each document separately
    for doc in raw_docs:
            print(f"\n=== FILE: {doc.metadata.get('source', 'Unknown')} ===")
            
            # Split this single document
            chunks = split_documents([doc])
            
            # Step 3: Show first 2 chunks
            for i, chunk in enumerate(chunks[:2]):  # only first 2 chunks
                print(f"\n--- CHUNK {i+1} ---")
                print(chunk.page_content[:300])  # first 300 characters
    else:
        print("No documents found to split.")