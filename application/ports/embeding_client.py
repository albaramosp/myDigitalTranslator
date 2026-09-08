from abc import ABC, abstractmethod
from typing import List, Optional

from domain.semantic_chunk import SemanticChunk


class EmbeddingClient(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[SemanticChunk]:
        ...