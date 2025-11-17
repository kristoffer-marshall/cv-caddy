"""
RAG (Retrieval Augmented Generation) operations for vector similarity and chunk retrieval.
"""
import numpy as np


def cosine_similarity(v1, v2):
    """
    Calculates cosine similarity between two numpy vectors.
    """
    # Ensure vectors are numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    
    return dot_product / (norm_v1 * norm_v2)


def estimate_tokens(text):
    """
    Estimates token count from text. Uses a simple approximation:
    ~4 characters per token for English text (conservative estimate).
    """
    if not text:
        return 0
    # Rough approximation: 1 token ≈ 4 characters for English
    # This is conservative to ensure we stay within limits
    return len(text) // 4


def find_relevant_chunks(query_embedding, all_embeddings, text_chunks, k, personal_info_chunk_indices=None):
    """
    Finds the top-k most similar chunks to the query.
    Always includes personal_info chunks to ensure important preferences/constraints are included.
    """
    if personal_info_chunk_indices is None:
        personal_info_chunk_indices = []
    
    similarities = []
    for emb in all_embeddings:
        sim = cosine_similarity(query_embedding, emb)
        similarities.append(sim)
    
    # Get the indices of the top-k most similar chunks
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    # Always include personal_info chunks (they contain important constraints/preferences)
    # Combine personal_info indices with top-k indices, removing duplicates
    combined_indices = list(set(top_k_indices.tolist() + personal_info_chunk_indices))
    
    # If we have more chunks than k, prioritize:
    # 1. Personal info chunks (always included)
    # 2. Top-k most similar chunks
    if len(combined_indices) > k:
        # Start with personal_info chunks
        result_indices = personal_info_chunk_indices.copy()
        
        # Add top-k chunks that aren't already in personal_info
        for idx in top_k_indices:
            if idx not in result_indices and len(result_indices) < k:
                result_indices.append(idx)
        
        # If we still have room and personal_info chunks, fill with remaining top-k
        remaining_slots = k - len(result_indices)
        if remaining_slots > 0:
            for idx in top_k_indices:
                if idx not in result_indices:
                    result_indices.append(idx)
                    remaining_slots -= 1
                    if remaining_slots == 0:
                        break
    else:
        result_indices = combined_indices
    
    # Return the text of those chunks
    return [text_chunks[i] for i in result_indices]

