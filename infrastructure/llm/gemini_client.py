from google import genai
from application.ports.llm_client import LlmClient
from dotenv import load_dotenv
import os

from domain.llm.llm_request import LlmRequest


class GeminiLlmClient(LlmClient):

    def __init__(self):
        load_dotenv()
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self._client = genai.Client()

    def generate(self, request: LlmRequest):
        response = self._client.models.generate_content(
            model="gemini-3-flash-preview",  # free model
            contents=request.user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=request.system_prompt
            ))

        return response.text