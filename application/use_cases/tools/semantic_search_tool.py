from application.ports.tool import Tool
from application.use_cases.semantic_search import SemanticSearch
from domain.tools import ToolDefinition


class SemanticSearchTool(Tool):
    def __init__(self,
                 semantic_search: SemanticSearch):
        self._semantic_search = semantic_search

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
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
        )

    def execute(self, **kwargs) -> str:
        return (self._semantic_search.
                semantic_search(kwargs['user_input'], kwargs['chunks']))
