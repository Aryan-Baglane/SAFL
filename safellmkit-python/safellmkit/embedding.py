from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingProvider:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        vec = self.model.encode([text], normalize_embeddings=True)[0]
        if isinstance(vec, np.ndarray):
            return vec.astype(np.float32).tolist()
        return list(map(float, vec))