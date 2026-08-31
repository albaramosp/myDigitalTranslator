import logging
from typing import Iterator

from google import genai
from application.ports.llm_client import LlmClient
from dotenv import load_dotenv
import os

from domain.llm.llm_request import LlmRequest
from domain.llm.llm_response import LlmResponse
from domain.llm.llm_response_chunk import LlmResponseChunk


class GeminiLlmClient(LlmClient):

    def __init__(self):
        load_dotenv()
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self._client = genai.Client()
        self.model = os.getenv("GEMINI_MODEL")

    def generate(self, request: LlmRequest):
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=request.user_prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=request.system_prompt,
                    response_mime_type="application/json"
                ))

            return LlmResponse(
                content=response.text,
                tokens=response.usage_metadata.total_token_count)

        except Exception as e:
            logger = logging.getLogger("myDigitalTranslator")
            logger.error(f"Gemini failed: {e}")

    def stream(self, request: LlmRequest) -> Iterator[LlmResponseChunk]:
        try:
            response = self._client.models.generate_content_stream(
            model=self.model,
            contents=request.user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=request.system_prompt,
                response_mime_type="application/json"
            ))

            for chunk in response:
                partial_text = chunk.text

                if partial_text:
                    yield LlmResponseChunk(
                        content=partial_text
                    )

        except Exception as e:
            logger = logging.getLogger("myDigitalTranslator")
            logger.error(f"Gemini failed: {e}")