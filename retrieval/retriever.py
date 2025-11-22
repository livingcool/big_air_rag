# retrieval/retriever.py

import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from embeddings.embedder import get_embeddings 
from embeddings.embedder import FAISS_INDEX_PATH, FAISS_METADATA_PATH, EMBEDDING_DIM
from numpy.linalg import norm

# --- Configuration ---
MAX_RETRIEVAL_CANDIDATES = 50 
FINAL_K_CHUNKS = 8 
MMR_LAMBDA = 0.7 

# --- Load Index and Metadata ---

def load_index_and_metadata() -> Tuple[Any, List[Dict[str, Any]]]:
    """Loads the pre-built Faiss index and the corresponding chunk metadata."""
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(FAISS_METADATA_PATH):
        print("CRITICAL: Faiss index or metadata not found. Run the ingestion pipeline first.")
        return None, []
    
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        return index, metadata
    except Exception as e:
        print(f"CRITICAL: Failed to load index/metadata: {e}")
        return None, []

# --- Retrieval Logic ---

def max_marginal_relevance(query_vector, candidates_vectors, candidates_metadata, k, lambda_val):
    """
    Performs Maximum Marginal Relevance (MMR) selection.
    """
    # --- CRITICAL FIX: Explicit check for NumPy array emptiness ---
    if candidates_vectors is None or len(candidates_vectors) == 0:
        return []

    # Calculate cosine similarity between vectors
    def cosine_similarity(v1, v2):
        # Ensure vectors are 1D for dot product
        v1 = v1.flatten() 
        v2 = v2.flatten()
        norm_v1 = norm(v1)
        norm_v2 = norm(v2)
        if norm_v1 == 0 or norm_v2 == 0: return 0.0
        return np.dot(v1, v2) / (norm_v1 * norm_v2)

    # Convert vectors to float32 for consistency
    query_vector = query_vector.astype('float32')
    candidates_vectors = candidates_vectors.astype('float32')
    
    # 1. Calculate Query-to-Document similarity (Relevance)
    query_similarity = [cosine_similarity(query_vector, v) for v in candidates_vectors]

    selected_indices = []
    
    for _ in range(min(k, len(candidates_vectors))):
        best_mmr_score = -float('inf')
        best_candidate_index = -1
        
        for i in range(len(candidates_vectors)):
            if i not in selected_indices:
                # 2. Calculate Document-to-Document similarity (Diversity)
                max_sim_to_selected = 0.0
                if selected_indices:
                    max_sim_to_selected = max(
                        cosine_similarity(candidates_vectors[i], candidates_vectors[j])
                        for j in selected_indices
                    )
                
                # 3. MMR Score
                mmr_score = lambda_val * query_similarity[i] - (1 - lambda_val) * max_sim_to_selected
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate_index = i
        
        if best_candidate_index != -1:
            selected_indices.append(best_candidate_index)

    return [candidates_metadata[i] for i in selected_indices]


def query_rag_system(query: str, doc_ids: List[str] = ["all"]) -> List[Dict[str, Any]]:
    """
    Main retrieval function: embeds query, searches index, and returns MMR-ranked chunks.
    """
    index, metadata = load_index_and_metadata()
    if index is None or not metadata: return []

    # 1. Embed the user query
    # Note: get_embeddings returns a list of lists.
    query_embedding_list = get_embeddings([{'source_text_snippet': query}]) 
    if not query_embedding_list: return []
    
    # Convert to NumPy array for Faiss
    query_vector = np.array(query_embedding_list[0], dtype='float32').reshape(1, -1)

    # 2. Approximate Nearest Neighbor (ANN) Search
    D, I = index.search(query_vector, MAX_RETRIEVAL_CANDIDATES)
    
    valid_indices = I[0][I[0] != -1]
    
    # 3. Prepare Candidates
    candidates_metadata = [metadata[int(i)] for i in valid_indices]
    
    # Retrieve vectors for MMR diversity check
    candidates_vectors = index.reconstruct_batch(valid_indices)

    # 4. Apply MMR Reranking
    print(f"Retrieval: Reranking {len(candidates_metadata)} candidates using MMR...")
    final_chunks = max_marginal_relevance(
        query_vector[0], 
        candidates_vectors,
        candidates_metadata,
        FINAL_K_CHUNKS,
        MMR_LAMBDA
    )

    return final_chunks

# --- Context Formatting ---

def format_context_for_llm(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Formats the retrieved chunks into a single string with mandatory citation tags.
    """
    context = []
    for chunk in retrieved_chunks:
        # Citation Tag Format: [DOC:ID | PAGE:NUMBER | CHUNK:ID]
        chunk_id_short = chunk['id'].split('-')[0]
        citation_tag = (
            f"[DOC:{chunk['doc_id']} | PAGE:{chunk['page']} | CHUNK:{chunk_id_short}]"
        )
        snippet = chunk['source_text_snippet'].replace('\n', ' ').strip()
        context.append(f"{citation_tag} {snippet}")
        
    return "\n\n".join(context)