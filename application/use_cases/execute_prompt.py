from application.ports.llm_client import LlmClient
from domain.prompts.prompts import Prompt
from domain.llm.llm_response import LlmResponse
import logging

class ExecutePrompt:

    def __init__(
            self,
            llm: LlmClient,
            prompt_builder: Prompt
    ):
        self._llm = llm
        self._prompt_builder = prompt_builder

    def execute(self, user_input: str) -> str:
        llm_request = self._prompt_builder.build(user_input)
        llm_response = self._llm.generate(llm_request)
        logger = logging.getLogger("mydigitaltranslator")
        logger.debug(f"Tokens: {llm_response.tokens}")
        return llm_response.content