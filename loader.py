# rag/loader.py

import os
from typing import List, Dict, Type
from langchain_community.document_loaders import (
    DirectoryLoader, 
    TextLoader, 
    PyPDFLoader, 
    UnstructuredMarkdownLoader
)
from langchain_core.documents import Document

# Mapping extensions to their respective LangChain loaders
LOADER_MAPPING: Dict[str, Type] = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".md": UnstructuredMarkdownLoader,
}

def load_documents(directory_path: str, extension: str = ".txt") -> List[Document]:
    """
    A generalized loader that handles different file types based on the extension provided.
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    # Select the loader class based on the extension
    loader_cls = LOADER_MAPPING.get(extension.lower())
    
    if not loader_cls:
        raise ValueError(f"Unsupported extension: {extension}. Supported: {list(LOADER_MAPPING.keys())}")

    print(f"--- Loading {extension} files from: {directory_path} ---")

    # The 'glob' now adapts to the extension provided
    loader = DirectoryLoader(
        directory_path,
        glob=f"**/*{extension}",
        loader_cls=loader_cls,
        show_progress=True,
        use_multithreading=True
    )

    try:
        documents = loader.load()
        
        # General Metadata Cleaning: Ensure source is just the filename
        for doc in documents:
            if "source" in doc.metadata:
                doc.metadata["source"] = os.path.basename(doc.metadata["source"])
        
        print(f"Successfully loaded {len(documents)} document(s).")
        return documents

    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

if __name__ == "__main__":
    # Test with default txt
    docs = load_documents("data", extension=".txt")



    print(f"\n{'='*20} CONTENT PREVIEW {'='*20}")
    for doc in docs:
        # Get first 5 lines
        lines = doc.page_content.splitlines()[:5]
        preview = "\n".join(lines) if lines else "[Empty File]"
        
        print(f"\n Source: {doc.metadata.get('source')}")
        print("-" * 30)
        print(preview)
        print("-" * 30)
    print(f"\n{'='*55}")