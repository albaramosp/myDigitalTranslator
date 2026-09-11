import unittest
from unittest import mock
from unittest.mock import Mock

from application.use_cases.tools.semantic_search_tool import SemanticSearchTool
from domain.tools import ToolDefinition


class SemanticSearchToolTests(unittest.TestCase):
    def setUp(self):
        self._semantic_search = Mock()
        self._semantic_search.semantic_search.return_value = "result"

        self._sut = SemanticSearchTool(self._semantic_search)

    def test_get_definition(self):
        result = self._sut.get_definition()
        self.assertEqual(
            ToolDefinition(
                name="execute",
                description="Performs a semantic search on the given chunks",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_input": {
                            "type": "string",
                            "description": "User query for the LLM model"
                        },
                        "chunks": {
                            "type": "array[SemanticChunk]",
                            "description": "vector of chunks"
                        }
                    },
                    "required": ["user_input", "chunks"]
                }
            ),
            result
        )

    def test_execute(self):
        result = self._sut.execute(**{
            "user_input": "a query",
            "chunks": []
        })

        self.assertEqual("result", result)
        self._semantic_search.semantic_search.assert_called_once_with(
            "a query",
            []
        )