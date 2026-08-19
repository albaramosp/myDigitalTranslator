import unittest
from unittest.mock import Mock

from application.use_cases.adapt_web_text import AdaptWebText
from domain.llm.llm_response import LlmResponse


class AdaptWebTextTests(unittest.TestCase):
    def setUp(self):
        self._mocked_llm_client = Mock()
        self._sut = AdaptWebText(
            self._mocked_llm_client
        )

    def test_adapt_web_text(self):
        self._mocked_llm_client.generate.side_effect = [
            LlmResponse(
                content='{"type": "news", "confidence": 0.95}',
                tokens=100
            ),
            LlmResponse(
                content='{"adapted_text": "A test news"}',
                tokens=150
            ),
            LlmResponse(
                content='{"content": "## A test news title \\n\\nA test news content."}',
                tokens=80
            ),
        ]

        result = self._sut.execute("Text")

        assert self._mocked_llm_client.generate.call_count == 3
        assert result == "## A test news title \n\nA test news content."

        calls = self._mocked_llm_client.generate.call_args_list

        assert calls[0].args[0].user_prompt == "Text"
        assert calls[1].args[0].user_prompt == "Text"
        assert calls[2].args[0].user_prompt == '{"adapted_text": "A test news"}'