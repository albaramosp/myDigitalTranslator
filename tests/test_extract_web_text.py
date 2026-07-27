import unittest
from unittest.mock import Mock
from application.use_cases.extract_web_text import ExtractWebText


class ExtractWebTextTests(unittest.TestCase):
    def setUp(self):
        self._mocked_web_adapter = Mock()
        self._mocked_web_downloader = Mock()
        self._sut = ExtractWebText(
            self._mocked_web_downloader,
            self._mocked_web_adapter,
        )

    def test_should_download_html_and_adapt_it(self):
        self._mocked_web_adapter.download.return_value = "<html>content</html>"
        self._mocked_web_adapter.adapt.return_value = "cleaned"

        use_case = ExtractWebText(
            downloader=self._mocked_web_adapter,
            adapter=self._mocked_web_adapter
        )

        result = use_case.execute("https://example.com")

        assert result == "cleaned"
        self._mocked_web_adapter.download.assert_called_once_with("https://example.com")
        self._mocked_web_adapter.adapt.assert_called_once_with("<html>content</html>")

    def test_should_return_none_when_download_fails(self):
        self._mocked_web_adapter.download.return_value = None

        use_case = ExtractWebText(
            downloader=self._mocked_web_adapter,
            adapter=self._mocked_web_adapter
        )

        result = use_case.execute("https://example.com")

        assert result is None
        self._mocked_web_adapter.download.assert_called_once_with("https://example.com")
        self._mocked_web_adapter.adapt.assert_not_called()


