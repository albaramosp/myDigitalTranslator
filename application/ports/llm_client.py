from abc import ABC, abstractmethod
from typing import Iterator, List, Optional

from domain.llm.llm_request import LlmRequest
from domain.llm.llm_response import LlmResponse
from domain.llm.llm_response_chunk import LlmResponseChunk
from domain.tools import ToolDefinition


class LlmClient(ABC):
    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, model: str):
        self._model = model

    @abstractmethod
    def generate(self, request: LlmRequest, tools: Optional[List[ToolDefinition]] = None) -> LlmResponse:
        pass

    @abstractmethod
    def stream(self, request: LlmRequest, tools: Optional[List[ToolDefinition]] = None) -> Iterator[LlmResponseChunk]:
        ...