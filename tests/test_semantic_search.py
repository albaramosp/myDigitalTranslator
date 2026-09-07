import unittest
from unittest.mock import Mock
import numpy as np
from application.use_cases.semantic_search import SemanticSearch
from domain.semantic_chunk import SemanticChunk


class SemanticSearchTests(unittest.TestCase):
    def setUp(self):
        self._mocked_embedding_client = Mock()
        self._sut = SemanticSearch(self._mocked_embedding_client)

    def test_semantic_search_returns_chunks_by_similarity(self):
        self._mocked_embedding_client.embed.return_value = [
            SemanticChunk(
                text="query",
                embedding=np.array([1.0, 0.0])
            )
        ]

        index = [
            SemanticChunk(
                text="Pizza",
                embedding=np.array([0.0, 1.0])
            ),
            SemanticChunk(
                text="Colosseum",
                embedding=np.array([0.9, 0.1])
            ),
            SemanticChunk(
                text="Rome",
                embedding=np.array([1.0, 0.0])
            ),
        ]

        result = self._sut.semantic_search(
            query="ancient Roman monument",
            index=index,
            top_k=3
        )

        assert result == [
            "Rome",
            "Colosseum",
            "Pizza"
        ]