from application.ports.llm_client import LlmClient
from domain.prompts.prompt_builder import PromptBuilder


class TranslateWebText:

    def __init__(
            self,
            llm: LlmClient,
            prompt_builder: PromptBuilder
    ):
        self._llm = llm
        self._prompt_builder = prompt_builder

    def execute(self, user_input: str) -> str:
        return self._llm.generate(self._prompt_builder.build(user_input))