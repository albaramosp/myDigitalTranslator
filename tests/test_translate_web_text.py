import unittest
from unittest.mock import Mock

from application.use_cases.translate_web_text import TranslateWebText
from domain.llm.llm_request import LlmRequest


class TranslateWebTextTests(unittest.TestCase):
    def setUp(self):
        self._mocked_llm_client = Mock()
        self._mocked_prompt_builder = Mock()
        self._sut = TranslateWebText(
            self._mocked_llm_client,
            self._mocked_prompt_builder,
        )


    def test_should_build_prompt_and_send_it_to_llm(self):
        request = LlmRequest(
            system_prompt="system",
            user_prompt="adapted"
        )

        self._mocked_prompt_builder.build.return_value = request
        self._mocked_llm_client.generate.return_value = "answer"

        result = self._sut.execute("original")

        # Assert
        assert result == "answer"

        self._mocked_prompt_builder.build.assert_called_once_with("original")
        self._mocked_llm_client.generate.assert_called_once_with(request)