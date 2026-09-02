import logging
from typing import Iterator

from litellm import completion
from application.ports.llm_client import LlmClient
from dotenv import load_dotenv
import os
from domain.llm.llm_request import LlmRequest
from domain.llm.llm_response import LlmResponse
from domain.llm.llm_response_chunk import LlmResponseChunk


class LiteLlmClient(LlmClient):

    def __init__(self):
        load_dotenv()
        self._groq_key = os.environ.get("GROQ_API_KEY")
        self._gemini_key = os.environ.get("GEMINI_API_KEY")
        self._groq_model = f"groq/{os.getenv('GROQ_MODEL')}"
        self._gemini_model = f"gemini/{os.getenv('GEMINI_MODEL')}"

    def generate(self, request: LlmRequest):
        try:
            rs = completion(
                messages=[
                    {
                        "role": "user",
                        "content": request.user_prompt
                    },
                    {
                        "role": "system",
                        "content": request.system_prompt,
                    }
                ],
                model=self._groq_model,
                fallbacks=[self._gemini_model],
                reasoning_effort="none",
                response_format={"type": "json_object"}
            )

            return LlmResponse(
                content=rs.choices[0].message.content,
                tokens=rs.usage.total_tokens
            )
        except Exception as e:
            logger = logging.getLogger("myDigitalTranslator")
            logger.error(f"All models failed: {e}")

    def stream(self, request: LlmRequest) -> Iterator[LlmResponseChunk]:
        try:
            rs = completion(
                messages=[
                    {
                        "role": "user",
                        "content": request.user_prompt
                    },
                    {
                        "role": "system",
                        "content": request.system_prompt,
                    }
                ],
                model=self._gemini_model,
                stream=True,
                fallbacks=[self._groq_model],
                reasoning_effort="none",
                response_format={"type": "json_object"}
            )

            for chunk in rs:
                partial_text = chunk.choices[0].delta.content

                if partial_text:
                    yield LlmResponseChunk(
                        content=partial_text
                    )
        except Exception as e:
            logger = logging.getLogger("myDigitalTranslator")
            logger.error(f"All models failed: {e}")