import numpy as np

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

