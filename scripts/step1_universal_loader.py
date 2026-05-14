import os
from langchain_community.document_loaders import (
    PyPDFLoader, 
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    TextLoader
)

def load_document(file_path):
    print(f"Loading file: {file_path}")
    
    # Extract the file extension
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    
    # Route to the appropriate loader based on the file extension
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
        # Bypass unconfigured formats and trigger a warning
        print(f"Warning: Unsupported file format ignored -> {extension}\n")
        return []
        
    # Load the document
    try:
        docs = loader.load()
        print(f"-> Successfully loaded. Extracted {len(docs)} document block(s).\n")
        return docs
    except Exception as e:
        print(f"-> Error loading {file_path}: {e}\n")
        return []

if __name__ == "__main__":
    # Dynamically resolve the absolute path of the current script's directory
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the absolute path to the data directory (one level up, then into 'data')
    data_dir = os.path.abspath(os.path.join(current_script_dir, "..", "data"))
    
    all_loaded_docs = []
    
    print("=== Starting Universal Document Loader ===\n")
    
    if os.path.exists(data_dir):
        # Iterate through all files in the data directory
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            
            # Skip directories and hidden files (like .DS_Store)
            if os.path.isfile(file_path) and not filename.startswith('.'):
                docs = load_document(file_path)
                if docs:
                    all_loaded_docs.extend(docs)
                    
        print("=== Final Summary ===")
        print(f"Total document blocks loaded from ALL files: {len(all_loaded_docs)}")
        
        # Optional: Print a tiny preview of the very first block to verify content
        if all_loaded_docs:
            print("\n--- Preview of the first loaded block ---")
            print(all_loaded_docs[0].page_content[:300] + "...\n")
            
    else:
        print(f"Error: Data directory not found at {data_dir}")