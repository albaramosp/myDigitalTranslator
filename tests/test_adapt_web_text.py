import unittest
from unittest.mock import Mock

from application.use_cases.adapt_single_web_text import AdaptSingleWebText
from domain.llm.llm_response import LlmResponse


class AdaptWebTextTests(unittest.TestCase):
    def setUp(self):
        self._mocked_llm_client = Mock()
        self._sut = AdaptSingleWebText(
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
            )
        ]

        result = self._sut.execute("Text")

        assert self._mocked_llm_client.generate.call_count == 2
        assert result == '{"adapted_text": "A test news"}'

        calls = self._mocked_llm_client.generate.call_args_list

        assert calls[0].args[0].user_prompt == "Text"
        assert calls[1].args[0].user_prompt == "Text"