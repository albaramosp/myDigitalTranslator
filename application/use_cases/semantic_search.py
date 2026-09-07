from application.ports.embeding_client import EmbeddingClient
import numpy as np

from domain.semantic_chunk import SemanticChunk


class SemanticSearch:
    def __init__(self, client: EmbeddingClient):
        self.client = client

    @staticmethod
    def _cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def semantic_search(
            self,
            query: str,
            index: list[SemanticChunk],
            top_k: int = 3,
    ):
        query_chunk = self.client.embed(query)

        scores = []

        for chunk in index:
            score = self._cosine_similarity(
                query_chunk[0].embedding,
                chunk.embedding
            )

            scores.append((score, chunk.text))

        scores.sort(reverse=True)

        return [
            text
            for _, text in scores[:top_k]
        ]
