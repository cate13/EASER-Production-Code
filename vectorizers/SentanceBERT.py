import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import Normalizer

class SBERTVectorMaker:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.normalizer = Normalizer(norm='l2', copy=False)

    def get_vector(self, text):
        # Encode returns a numpy array by default
        embedding = self.model.encode([text])
        
        # Normalize to unit length (similar to your previous pipeline)
        normalized_embedding = self.normalizer.transform(embedding)
        
        return normalized_embedding[0]