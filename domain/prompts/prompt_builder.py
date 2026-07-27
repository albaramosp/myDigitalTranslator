from abc import ABC, abstractmethod
from domain.llm.llm_request import LlmRequest


class PromptBuilder(ABC):

    @staticmethod
    @abstractmethod
    def build(text) -> LlmRequest:
        ...