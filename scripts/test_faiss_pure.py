"""
FAISS compatibility note (2026-05-31)
======================================
This script verifies that FAISS can create an index, insert vectors,
run nearest-neighbor search, persist to disk, and reload — the core
operations needed for a RAG vector store.

Environment: Windows 11 (10.0.26200), Intel Kaby Lake/R generation,
              Python 3.12.9, faiss-cpu 1.14.2, numpy 1.26.4

All phases pass: IndexFlatL2(768) creation, vector add, search (distance
~0), faiss.write_index / faiss.read_index round-trip.

FAISS was adopted as the vector store after ChromaDB's Rust engine
proved incompatible with this machine (see test_chroma_pure.py).
==============================================================
"""

import os
import sys
import numpy as np
import faiss

def test_faiss_write():
    sys.stdout.reconfigure(line_buffering=True)

    print("\n=== Phase 1: Environment Setup ===")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.abspath(os.path.join(current_dir, "..", "test_faiss_db"))
    os.makedirs(db_dir, exist_ok=True)

    index_path = os.path.join(db_dir, "test_index.faiss")
    print(f"Index path: {index_path}")

    print("\n=== Phase 2: Create Index ===")
    dim = 768  # Gemini embedding dimension
    index = faiss.IndexFlatL2(dim)
    print(f"[OK] FAISS IndexFlatL2 created. Dimension: {dim}")

    print("\n=== Phase 3: Add Vectors ===")
    dummy_vector = np.array([[0.1] * dim], dtype=np.float32)
    print(f"Vector shape: {dummy_vector.shape}, dtype: {dummy_vector.dtype}")
    index.add(dummy_vector)
    print(f"[OK] 1 vector added. Index total: {index.ntotal}")

    print("\n=== Phase 4: Search (verify read) ===")
    query = np.array([[0.1] * dim], dtype=np.float32)
    distances, indices = index.search(query, k=1)
    print(f"Nearest neighbor distance: {distances[0][0]:.6f}")
    print(f"Nearest neighbor index: {indices[0][0]}")

    if indices[0][0] == 0 and distances[0][0] < 1e-6:
        print("[OK] Query returned the inserted vector (distance ~0).")
    else:
        print("[FAIL] Query did not return the expected result.")
        sys.exit(1)

    print("\n=== Phase 5: Persist to Disk ===")
    faiss.write_index(index, index_path)
    file_size = os.path.getsize(index_path)
    print(f"[OK] Index written to disk. File size: {file_size} bytes")

    print("\n=== Phase 6: Load from Disk ===")
    loaded = faiss.read_index(index_path)
    print(f"[OK] Index loaded. Total vectors: {loaded.ntotal}")

    if loaded.ntotal == 1:
        print("\n[SUCCESS] FAISS is fully functional on this system.")
    else:
        print("\n[WARN] Loaded index has unexpected count.")
        sys.exit(1)

if __name__ == "__main__":
    test_faiss_write()
