# app/vector/embedding_service.py
# Centralized embedding service to compute reusable embeddings for query variants.

from typing import List
from chromadb.utils import embedding_functions

class EmbeddingService:
    def __init__(self, model_name: str = "default"):
        # Use the same default embedding the collection used, but centralized here
        self._ef = embedding_functions.DefaultEmbeddingFunction()

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Deterministic embedding for a batch of texts
        return self._ef.__call__(texts)
