import os

from dotenv import load_dotenv

from application.ports.llm_client import LlmClient
from infrastructure.llm.gemini_client import GeminiLlmClient
from infrastructure.llm.groq_client import GroqLlmClient
from infrastructure.llm.local_client import LlamaClient

class LlmFactory:

    @staticmethod
    def create() -> LlmClient:
        load_dotenv()
        provider = os.environ.get("LLM_PROVIDER", "llama").lower()

        if provider == "gemini":
            return GeminiLlmClient()
        elif provider == "groq":
            return GroqLlmClient()
        elif provider == "llama":
            return LlamaClient()

        raise ValueError(f"Unknown LLM provider: {provider}")