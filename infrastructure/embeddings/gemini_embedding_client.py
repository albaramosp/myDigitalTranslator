import os

from dotenv import load_dotenv
from google import genai

from application.ports.embeding_client import EmbeddingClient
from typing import List
import numpy as np

from domain.semantic_chunk import SemanticChunk


class GeminiEmbeddingClient(EmbeddingClient):
    @staticmethod
    def chunk_text(
            text: str,
            chunk_size: int = 500,
            overlap: int = 100) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            # Move the window forward, but stay back by the 'overlap' amount
            start += (chunk_size - overlap)
        return chunks

    def embed(self, text: str) -> List[SemanticChunk]:
        load_dotenv()
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        client = genai.Client()

        chunks = self.chunk_text(text)
        embedded_content = client.models.embed_content(
            model="gemini-embedding-2",
            contents=text
        )
        if not embedded_content.embeddings:
            return []

        vectors = embedded_content.embeddings[0].values

        return [
            SemanticChunk(text=t, embedding=np.array(v))
            for t, v in zip(chunks, vectors)
        ] if vectors else []