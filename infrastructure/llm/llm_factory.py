import os

from dotenv import load_dotenv

from application.ports.llm_client import LlmClient
from infrastructure.llm.gemini_client import GeminiLlmClient
from infrastructure.llm.groq_client import GroqLlmClient
from infrastructure.llm.lite_llm_client import LiteLlmClient
from infrastructure.llm.ollama_client import OllamaClient


class LlmFactory:

    @staticmethod
    def create() -> LlmClient:
        load_dotenv()
        provider = os.environ.get("LLM_PROVIDER", "").lower()

        if provider == "gemini":
            return GeminiLlmClient()
        elif provider == "groq":
            return GroqLlmClient()
        elif provider == "lite_llm":
            return LiteLlmClient()

        raise ValueError(f"Unknown LLM provider: {provider}")