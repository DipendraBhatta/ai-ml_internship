# main.py
import sys
from rag.qa_chain import get_qa_chain

def start_app():
    print("\n" + "="*50)
    print(" RAG CHATBOT INITIALIZED (Powered by Groq & Llama-3)")
    print("="*50)
    print("Type 'exit' or 'quit' to stop the session.\n")

    # Initialize the chain once
    try:
        qa_chain = get_qa_chain()
    except Exception as e:
        print(f" Initialization Error: {e}")
        return

    while True:
        query = input("👤 You: ")
        
        if query.lower() in ["exit", "quit", "q"]:
            print("\n👋 Goodbye!")
            break
        
        if not query.strip():
            continue

        try:
            print(" AI is thinking...")
            # Use the modern .invoke() method
            response = qa_chain.invoke({"query": query})
            
            print(f"\n✨ Response: {response['result']}")
            
            # Show sources for transparency
            if response.get("source_documents"):
                sources = set(doc.metadata.get("source") for doc in response["source_documents"])
                print(f"Sources: {', '.join(sources)}")
            print("-" * 30)

        except Exception as e:
            print(f" Error during chat: {e}")

if __name__ == "__main__":
    start_app()