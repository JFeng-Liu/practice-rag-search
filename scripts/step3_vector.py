import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader, UnstructuredExcelLoader, TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


def get_document_chunks(data_dir):
    """Pipeline to load and chunk documents from the specified directory."""
    all_docs = []

    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            if not os.path.isfile(file_path) or filename.startswith('.'):
                continue

            _, ext = os.path.splitext(file_path)
            ext = ext.lower()

            if ext == '.pdf': loader = PyPDFLoader(file_path)
            elif ext in ['.docx', '.doc']: loader = UnstructuredWordDocumentLoader(file_path)
            elif ext in ['.pptx', '.ppt']: loader = UnstructuredPowerPointLoader(file_path)
            elif ext in ['.xlsx', '.xls']: loader = UnstructuredExcelLoader(file_path)
            elif ext == '.txt': loader = TextLoader(file_path, encoding='utf-8')
            else: continue

            try:
                all_docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)

    for chunk in chunks:
        clean_metadata = {}
        if 'source' in chunk.metadata: clean_metadata['source'] = chunk.metadata['source']
        if 'page' in chunk.metadata: clean_metadata['page'] = chunk.metadata['page']
        chunk.metadata = clean_metadata

    return chunks


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, "..", "data"))
    db_dir = os.path.abspath(os.path.join(current_dir, "..", "faiss_db"))

    print("=== Phase 1: Loading & Chunking ===")
    chunks = get_document_chunks(data_dir)
    print(f"-> Total chunks to process: {len(chunks)}\n")

    if not chunks:
        print("Error: No chunks generated.")
        sys.exit(1)

    print("=== Phase 2: Embedding & Indexing ===")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    vectorstore = FAISS.from_documents(chunks, embeddings)

    print(f"=== Phase 3: Saving to Disk ===")
    vectorstore.save_local(db_dir)
    print(f"[OK] Saved to: {db_dir}")

    print(f"\n[SUCCESS] Vector database built. Total documents: {vectorstore.index.ntotal}")
