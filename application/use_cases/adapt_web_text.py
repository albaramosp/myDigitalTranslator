from application.use_cases.execute_prompt import ExecutePrompt
from domain.prompts.prompts import PromptFactory, TextClassificationPrompt, FormattingPrompt
from application.ports.llm_client import LlmClient
import json


class AdaptWebText:
    def __init__(self, llm: LlmClient):
        self._llm = llm

    def execute(self, text):
        classification_prompt = TextClassificationPrompt()
        classification = classification_prompt.parse(
            ExecutePrompt(
                self._llm,
                classification_prompt
            ).execute(text)
        )

        adaptation_prompt = PromptFactory.create(classification.type)

        adapted_text = ExecutePrompt(
            self._llm,
            adaptation_prompt
        ).execute(text)

        formatting_prompt = FormattingPrompt() 
        formatted = ExecutePrompt(
            self._llm,
           formatting_prompt
        ).execute(adaptation_prompt.parse(adapted_text))

        return formatting_prompt.parse(formatted).content