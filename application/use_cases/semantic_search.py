from application.ports.embeding_client import EmbeddingClient
import numpy as np
from application.ports.llm_client import LlmClient
from application.use_cases.execute_prompt import ExecutePrompt
from domain.prompts import ContextSummaryPrompt
from domain.semantic_chunk import SemanticChunk


class SemanticSearch:
    def __init__(self,
                 embedding_client: EmbeddingClient,
                 llm_client: LlmClient):
        self._embedding_client = embedding_client
        self._llm_client = llm_client

    @staticmethod
    def _cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def semantic_search(
            self,
            query: str,
            index: list[SemanticChunk],
            top_k: int = 3,
    ):
        query_chunk = self._embedding_client.embed(query)

        scores = []

        for chunk in index:
            score = self._cosine_similarity(
                query_chunk[0].embedding,
                chunk.embedding
            )

            scores.append((score, chunk.text))

        scores.sort(reverse=True)

        relevant_chunks = [
            text
            for _, text in scores[:top_k]
        ]

        context_block = "\n---\n".join(relevant_chunks)
        compound_query = f"Context: {context_block}\n User query: {query}"
        prompt = ContextSummaryPrompt()

        adapted_text = ExecutePrompt(
            self._llm_client,
            prompt
        ).execute(compound_query)

        return prompt.parse(
            adapted_text
        )
