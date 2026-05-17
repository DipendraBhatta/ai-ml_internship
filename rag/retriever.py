# rag/retriever.py
from langchain_chroma import Chroma
from rag.embeddings import get_embedding_model

def get_retriever(db_path="./chroma_db", k=3):
    """
    Returns a LangChain-compatible retriever object.
    """
    # 1. Load embedding model
    embeddings = get_embedding_model()

    # 2. Load Chroma DB
    vector_db = Chroma(
        persist_directory=db_path, 
        embedding_function=embeddings
    )

    # 3. FIX: Return the formal LangChain retriever object
    # This object has the methods RetrievalQA needs to work.
    return vector_db.as_retriever(search_kwargs={"k": k})

# ----- Test block -----
if __name__ == "__main__":
    retriever_obj = get_retriever()
    query = "Tell me something about Google"
    
    # In LangChain, you call .invoke() on the retriever object
    results = retriever_obj.invoke(query)

    print(f"\nTop {len(results)} chunks for query: '{query}'\n")
    for i, chunk in enumerate(results):
        print(f"--- Chunk {i+1} ---")
        print(chunk.page_content[:300], "\n")