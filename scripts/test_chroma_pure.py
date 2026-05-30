"""
ChromaDB compatibility note (2026-05-31)
=========================================
This script isolates collection.add() to verify whether ChromaDB can
persist vectors on this machine. Under the tested environment, the call
terminates the process with a native access violation inside the Rust
segment writer:

    Windows fatal exception: access violation
    chromadb/api/rust.py:484 in _add

The crash is independent of client mode — PersistentClient,
EphemeralClient, and the standalone chroma server all trigger the same
fault at the Rust FFI boundary during the add operation.

Environment: Windows 11 (10.0.26200), Intel Kaby Lake/R generation,
              Python 3.12.9, SQLite 3.45.3

Mitigations attempted without effect:
  - chromadb 1.5.9, 1.0.21, 0.5.23 (0.5.x blocked by missing MSVC toolchain)
  - pysqlite3 override (system SQLite already meets the 3.35.0 floor)
  - numpy 1.26.x and 2.4.x
  - onnxruntime reinstall

Outcome: switched vector storage to FAISS. This file remains as a
         record of the ChromaDB investigation.
==============================================================
"""

import os
import sys
import uuid

try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
    print("[OK] Successfully hijacked sqlite3 with pysqlite3-binary.")
except ImportError:
    print("[WARN] pysqlite3 not found. Using system default sqlite3. (May cause crashes)")
# -----------------------------------------

import chromadb

def test_chroma_write():
    # Force stdout flush
    sys.stdout.reconfigure(line_buffering=True)

    print("\n=== Phase 1: Environment Setup ===")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.abspath(os.path.join(current_dir, "..", "test_chroma_db"))

    # Clean start
    if os.path.exists(db_dir):
        print(f"Removing old test DB at {db_dir}...")
        import shutil
        shutil.rmtree(db_dir, ignore_errors=True)

    print(f"DB Path: {db_dir}")

    print("\n=== Phase 2: Client Initialization ===")
    try:
        client = chromadb.PersistentClient(path=db_dir)
        collection = client.get_or_create_collection(name="test_collection")
        print("[OK] Chroma client and collection initialized successfully.")
    except Exception as e:
        print(f"[FAIL] Failed to initialize Chroma: {e}")
        sys.exit(1)

    print("\n=== Phase 3: Mock Data Generation ===")
    # Create fake data. Gemini embedding dimension is 768.
    # We create a simple dummy vector with 768 floats.
    dummy_text = "This is a test document to verify ChromaDB write operations."
    dummy_vector = [0.1] * 768
    dummy_id = str(uuid.uuid4())
    dummy_metadata = {"source": "test", "page": 1}
    print("[OK] Mock data generated.")

    print("\n=== Phase 4: The Danger Zone (collection.add) ===")
    print("Attempting to write 1 document to disk...")

    try:
        collection.add(
            ids=[dummy_id],
            embeddings=[dummy_vector],
            documents=[dummy_text],
            metadatas=[dummy_metadata]
        )
        print("[OK] collection.add() executed WITHOUT crashing!")
    except Exception as e:
        print(f"[FAIL] CRASH CAUGHT inside collection.add(): {e}")
        sys.exit(1)

    print("\n=== Phase 5: Verification ===")
    try:
        count = collection.count()
        print(f"Total documents in DB: {count}")
        if count == 1:
            print("\n[SUCCESS] ChromaDB is fully functional on your system.")
        else:
            print("\n[WARN] Wrote without crashing, but count is not 1.")
    except Exception as e:
        print(f"[FAIL] Failed to count documents: {e}")

if __name__ == "__main__":
    test_chroma_write()
