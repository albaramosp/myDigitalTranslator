import logging
from litellm import completion
from application.ports.llm_client import LlmClient
from dotenv import load_dotenv
import os
from domain.llm.llm_request import LlmRequest
from domain.llm.llm_response import LlmResponse


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
                fallbacks=[self._gemini_model]
            )

            return LlmResponse(
                content=rs.choices[0].message.content,
                tokens=rs.usage.total_tokens
            )
        except Exception as e:
            logger = logging.getLogger("myDigitalTranslator")
            logger.error(f"All models failed: {e}")
