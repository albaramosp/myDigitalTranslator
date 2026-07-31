from groq import Groq
from application.ports.llm_client import LlmClient
from dotenv import load_dotenv
import os

from domain.llm.llm_request import LlmRequest


class GroqLlmClient(LlmClient):

    def __init__(self):
        load_dotenv()
        self._client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )

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
            model="llama-3.3-70b-versatile",
        )

        return chat_completion.choices[0].message.content