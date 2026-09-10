import unittest
from unittest.mock import Mock
import numpy as np
from application.use_cases.semantic_search import SemanticSearch
from domain.llm.llm_response import LlmResponse
from domain.semantic_chunk import SemanticChunk


class SemanticSearchTests(unittest.TestCase):
    def setUp(self):
        self._mocked_embedding_client = Mock()
        self._mocked_llm_client = Mock()
        self._sut = SemanticSearch(
            self._mocked_embedding_client,
            self._mocked_llm_client)

    def test_semantic_search_creates_chunks_by_similarity(self):
        self._mocked_embedding_client.embed.return_value = [
            SemanticChunk(
                text="query",
                embedding=np.array([1.0, 0.0])
            )
        ]

        self._mocked_llm_client.generate.return_value = LlmResponse(
            content='{"content": "my answer"}',
            tokens=5
        )

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

        user_query = "ancient Roman monument"

        result = self._sut.semantic_search(
            query=user_query,
            index=index,
            top_k=3
        )

        expected_context_block = "\n---\n".join([
            "Rome",
            "Colosseum",
            "Pizza"
        ])

        calls = self._mocked_llm_client.generate.call_args_list
        self.assertEqual(
            f"Context: {expected_context_block}\n User query: {user_query}",
            calls[0].args[0].user_prompt)

        self.assertEqual('my answer', result)
