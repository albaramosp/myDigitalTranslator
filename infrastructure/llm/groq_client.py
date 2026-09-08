import logging
from typing import Iterator

from groq import Groq
from application.ports.llm_client import LlmClient
from dotenv import load_dotenv
import os

from domain.llm.llm_request import LlmRequest
from domain.llm.llm_response import LlmResponse
from domain.llm.llm_response_chunk import LlmResponseChunk


class GroqLlmClient(LlmClient):

    def __init__(self):
        load_dotenv()
        self._client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        self.model = os.getenv("GROQ_MODEL")

    def generate(self, request: LlmRequest):
        try:
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

        except Exception as e:
            logger = logging.getLogger("myDigitalTranslator")
            logger.error(f"Groq failed: {e}")

    def stream(self, request: LlmRequest) -> Iterator[LlmResponseChunk]:
        try:
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
                response_format={"type": "json_object"} if request.json_output else None,
                stream=True
            )

            for chunk in chat_completion:
                partial_text = chunk.choices[0].delta.content

                if partial_text:
                    yield LlmResponseChunk(
                        content=partial_text
                    )

        except Exception as e:
            logger = logging.getLogger("myDigitalTranslator")
            logger.error(f"Groq failed: {e}")