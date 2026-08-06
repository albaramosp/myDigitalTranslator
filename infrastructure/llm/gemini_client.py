from google import genai
from application.ports.llm_client import LlmClient
from dotenv import load_dotenv
import os

from domain.llm.llm_request import LlmRequest
from domain.llm.llm_response import LlmResponse


class GeminiLlmClient(LlmClient):

    def __init__(self):
        load_dotenv()
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self._client = genai.Client()
        self.model = "gemini-3-flash-preview"

    def generate(self, request: LlmRequest):
        response = self._client.models.generate_content(
            model=self.model,
            contents=request.user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=request.system_prompt
            ))

        return LlmResponse(
            content=response.text,
            tokens=response.usage_metadata.total_token_count)   