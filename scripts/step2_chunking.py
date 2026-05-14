import os
from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader, UnstructuredExcelLoader, TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_document(file_path):
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    
    if extension == '.pdf':
        loader = PyPDFLoader(file_path)
    elif extension in ['.docx', '.doc']:
        loader = UnstructuredWordDocumentLoader(file_path)
    elif extension in ['.pptx', '.ppt']:
        loader = UnstructuredPowerPointLoader(file_path)
    elif extension in ['.xlsx', '.xls']:
        loader = UnstructuredExcelLoader(file_path)
    elif extension == '.txt':
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        return []
        
    try:
        return loader.load()
    except Exception:
        return []

def chunk_documents(docs):
    # chunk_size: Maximum number of characters per chunk
    # chunk_overlap: Number of overlapping characters between adjacent chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    return chunks

if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_script_dir, "..", "data"))
    
    all_loaded_docs = []
    
    print("=== Step 1: Loading Documents ===")
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            if os.path.isfile(file_path) and not filename.startswith('.'):
                docs = load_document(file_path)
                if docs:
                    all_loaded_docs.extend(docs)
        print(f"Total original document blocks loaded: {len(all_loaded_docs)}\n")
        
        print("=== Step 2: Chunking Documents ===")
        all_chunks = chunk_documents(all_loaded_docs)
        print(f"Total chunks created: {len(all_chunks)}")
        
        if all_chunks:
            print("\n--- Metadata of the first chunk ---")
            print(all_chunks[0].metadata)
            print("\n--- Content of the first chunk ---")
            print(all_chunks[0].page_content[:300] + "...")
    else:
        print(f"Error: Data directory not found at {data_dir}")