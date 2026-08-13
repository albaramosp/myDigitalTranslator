import unittest
from unittest.mock import Mock

from application.use_cases.adapt_web_text import AdaptWebText


class AdaptWebTextTests(unittest.TestCase):
    def setUp(self):
        self._mocked_llm_client = Mock()
        self._sut = AdaptWebText(
            self._mocked_llm_client
        )


    def test_llm(self):
        assert 2 == 2