# rag/qa_chain.py
import os
from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
from rag.retriever import get_retriever # Import the function

load_dotenv()

def get_groq_llm():
    api_key = os.getenv("GROQ_API_KEY")
    # Cleaned up parameters to stop Pydantic warnings
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.0,
        max_tokens=512
    )

def get_qa_chain():
    # 1. Initialize Groq LLM
    llm = get_groq_llm()

    # 2. GET THE RETRIEVER
    # CRITICAL: You MUST use the () here to CALL the function.
    # If you just write 'retriever = get_retriever', it fails.
    my_retriever = get_retriever() 

    # 3. Create the Classic RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=my_retriever, # Pass the object we just created
        return_source_documents=True
    )

    return qa_chain

# if __name__ == "__main__":
#     print("\n--- Starting Classic RAG Chain ---")
    
#     # Initialize
#     qa = get_qa_chain()
    
#     # Test Query
#     query = "where is the headquarter of Microsoft"
    
#     try:
#         # Use invoke for modern compatibility within the classic chain
#         result = qa.invoke({"query": query})
        
#         print("\n" + "="*50)
#         print(f"ANSWER: {result['result']}")
#         print("="*50)
        
#     except Exception as e:
#         print(f"Error: {e}")