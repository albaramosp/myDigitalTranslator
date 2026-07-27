from abc import ABC, abstractmethod
from domain.llm.llm_request import LlmRequest


class LlmClient(ABC):
    @abstractmethod
    def generate(self, request: LlmRequest) -> str:
        pass