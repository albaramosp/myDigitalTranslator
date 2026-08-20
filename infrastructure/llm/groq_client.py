from groq import Groq
from application.ports.llm_client import LlmClient
from dotenv import load_dotenv
import os

from domain.llm.llm_request import LlmRequest
from domain.llm.llm_response import LlmResponse


class GroqLlmClient(LlmClient):

    def __init__(self):
        load_dotenv()
        self._client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        self.model = os.getenv("GROQ_MODEL")

    def generate(self, request: LlmRequest):
        chat_completion = self._client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": request.user_prompt,
                },
                {
                    "role": "system",
                    "content": request.system_prompt,
                }
            ],
            model=self.model,
            response_format={"type": "json_object"}
        )

        return LlmResponse(
            content=chat_completion.choices[0].message.content,
            tokens=chat_completion.usage.total_tokens
        )