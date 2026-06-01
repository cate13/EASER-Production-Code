import numpy as np
from typing import List, Union

def combine_using_bilinear_pool(e_vec, t_vec):
    if isinstance(e_vec, dict):
        e_vec = list(e_vec.values())
    if isinstance(t_vec, dict):
        t_vec = list(t_vec.values())
    
    vec_a = np.array(e_vec)
    vec_a = np.append(vec_a, 1.0)
    vec_b = np.array(t_vec)
    vec_b = np.append(vec_b, 1.0)

    outer_product = np.outer(vec_a, vec_b)
    flattened = outer_product.flatten()
    normalized = np.sign(flattened) * np.sqrt(np.abs(flattened))
    norm = np.linalg.norm(normalized)

    if norm > 0:
        final_vector = normalized / norm
    else:
        final_vector = normalized
        
    return final_vector

def average_vectors(vectors: List[Union[List[float], dict]]):
    """
    Averages a list of vectors.
    
    Works for:
        - List[float]  (e.g., tf_idf)
        - Dict[str, float] (e.g., emotion_intensity, emotion, empath)

    Returns:
        Averaged vector (same structure as input)
    """

    if not vectors:
        raise ValueError("Vector list is empty.")
    
    vectors = [v for v in vectors if v is not None]

    first = vectors[0]

    # Case 1: List-based vector (e.g., tf_idf)
    if isinstance(first, list):
        return np.mean(np.array(vectors), axis=0).tolist()

    # Case 2: NumPy-based vector
    elif isinstance(first, np.ndarray):
        # We average along the rows (axis 0)
        return np.mean(np.array(vectors), axis=0)
    
    # Case 3: Dict-based vector (emotion/empath)
    elif isinstance(first, dict):
        keys = first.keys()

        # Safety check
        if not all(v.keys() == keys for v in vectors):
            raise ValueError("All dict vectors must have same keys.")

        return {
            key: sum(v[key] for v in vectors) / len(vectors)
            for key in keys
        }

    else:
        raise TypeError(f"Unsupported vector type. {vectors}")