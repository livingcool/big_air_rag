# embeddings/embedder.py (LOCAL FALLBACK EDITION)

import os
from typing import List, Dict, Any
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Configuration (GLOBAL CONSTANTS) ---
# We switch to a highly efficient local model
LOCAL_MODEL_NAME = "all-MiniLM-L6-v2" 
EMBEDDING_DIM = 384 # Dimension for all-MiniLM-L6-v2 is 384 (vs 3072 for OpenAI)

FAISS_INDEX_PATH = "data/processed/multimodal_rag.faiss"
FAISS_METADATA_PATH = "data/processed/multimodal_rag_metadata.json"

# --- Model Initialization ---
try:
    print(f"Embedder: Loading local model '{LOCAL_MODEL_NAME}'...")
    # This downloads the model once (approx 80MB) and runs locally
    LOCAL_CLIENT = SentenceTransformer(LOCAL_MODEL_NAME)
    print(f"Embedder: Local model initialized on device: {LOCAL_CLIENT.device}")

except Exception as e:
    print(f"CRITICAL: Failed to load local embedding model. Error: {e}")
    LOCAL_CLIENT = None


def get_embeddings(chunks: List[Dict[str, Any]]) -> List[List[float]]:
    """
    Generates embeddings locally using SentenceTransformers.
    """
    if not LOCAL_CLIENT:
        print("Embeddings failed: Local client not available.")
        return []

    texts = [chunk['source_text_snippet'] for chunk in chunks]
    
    try:
        # Encode locally
        embeddings = LOCAL_CLIENT.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
        
    except Exception as e:
        print(f"ERROR generating local embeddings: {e}")
        return []


def index_chunks(chunks: List[Dict[str, Any]]):
    """
    Generates embeddings and builds the Faiss index for the retrieval system.
    """
    if not chunks:
        print("No chunks provided for indexing.")
        return

    print(f"Indexing: Generating {len(chunks)} embeddings locally...")
    embeddings_list = get_embeddings(chunks)
    
    if not embeddings_list or len(embeddings_list) != len(chunks):
        print("Indexing failed due to incomplete or missing embeddings.")
        return

    # Faiss requires numpy array of float32
    vectors = np.array(embeddings_list, dtype='float32')
    
    # Create Index (L2 Distance)
    # Note: L2 is fine for normalized vectors, but Inner Product (IP) is better for cosine similarity.
    # For MVP simplicity, we stick to IndexFlatL2.
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(vectors)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    
    faiss.write_index(index, FAISS_INDEX_PATH)
    
    # Store metadata
    metadata = [
        {
            "id": chunk['id'],
            "doc_id": chunk['doc_id'],
            "page": chunk['page'],
            "modality": chunk['modality'],
            "bbox": chunk['bbox'],
            "source_text_snippet": chunk['source_text_snippet'] 
        }
        for chunk in chunks
    ]
    
    import json
    with open(FAISS_METADATA_PATH, 'w') as f:
        json.dump(metadata, f)

    print(f"Indexing Complete. Faiss index saved to {FAISS_INDEX_PATH}")
    print(f"Total indexed vectors: {index.ntotal}") 