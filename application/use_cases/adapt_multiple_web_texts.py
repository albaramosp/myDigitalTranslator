from application.use_cases.adapt_single_web_text import AdaptSingleWebText
from application.use_cases.execute_prompt import ExecutePrompt
from domain.prompts.prompts import FormattingPrompt, SynthesizePrompt, Response
from application.ports.llm_client import LlmClient


class AdaptMultipleWebTexts:
    def __init__(self, llm: LlmClient):
        self._llm = llm

    def pipeline(self, texts: list[str]) -> Response:
        adapted_texts = []

        adapter = AdaptSingleWebText(self._llm)

        for text in texts:
            adapted_texts.append(
                adapter.execute(text)
            )

        combined_text = "\n\n".join(adapted_texts)

        synthesis_prompt = SynthesizePrompt()

        return synthesis_prompt.parse(
            ExecutePrompt(
                self._llm,
                synthesis_prompt
            ).execute(combined_text)
        )

    def format(self, text: str):
        formatting_prompt = FormattingPrompt()

        return formatting_prompt.parse(
            ExecutePrompt(
                self._llm,
                formatting_prompt
            ).execute(text)
        )

    def format_streaming(self, text: str):
        formatting_prompt = FormattingPrompt()

        for chunk in ExecutePrompt(
                self._llm,
                formatting_prompt
        ).stream(text):
            yield chunk

    def stream(self, texts: list[str]):
        partial_txt = ""
        for chunk in self.format_streaming(self.pipeline(texts).content):
            partial_txt += chunk.content
            yield partial_txt

    def execute(self, texts: list[str]) -> str:
        return self.format(self.pipeline(texts).content).content
