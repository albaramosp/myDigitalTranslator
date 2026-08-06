from abc import ABC, abstractmethod
from domain.llm.llm_request import LlmRequest
from domain.llm.llm_response import LlmResponse

class LlmClient(ABC):
    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, model: str):
        self._model = model

    @abstractmethod
    def generate(self, request: LlmRequest) -> LlmResponse:
        pass