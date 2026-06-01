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

def cosine_similarity(vec1, vec2):
    if type(vec1) is not type(vec2):
        raise ValueError("Both vectors must be the same type.")

    # Case 1: List vectors (e.g., tf_idf)
    if isinstance(vec1, list):
        if len(vec1) != len(vec2):
            raise ValueError("Both list vectors must have same length.")

        v1 = np.array(vec1, dtype=float)
        v2 = np.array(vec2, dtype=float)

    # Case 2: Dict vectors (e.g., emotion/empath)
    elif isinstance(vec1, dict):
        if set(vec1.keys()) != set(vec2.keys()):
            raise ValueError("Both dict vectors must have same keys.")

        # Ensure consistent ordering
        keys = sorted(vec1.keys())
        v1 = np.array([vec1[k] for k in keys], dtype=float)
        v2 = np.array([vec2[k] for k in keys], dtype=float)
    elif isinstance(vec1, np.ndarray):
        if vec1.shape != vec2.shape:
            raise ValueError("Both numpy vectors must have the same shape.")
        v1 = vec1
        v2 = vec2
    else:
        raise TypeError(f"Unsupported vector type. {vec1} {vec2}")

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(v1, v2) / (norm1 * norm2))