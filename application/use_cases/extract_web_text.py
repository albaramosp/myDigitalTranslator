from typing import Optional

from application.adapters.web_adapter import WebAdapter
from application.ports.web_downloader import WebDownloader


class ExtractWebText:

    def __init__(
        self,
        downloader: WebDownloader,
        adapter: WebAdapter,
    ):
        self.downloader = downloader
        self.extractor = adapter

    def execute(self, url: str) -> Optional[str]:

        html = self.downloader.download(url)

        if html is None:
            return None

        text = self.extractor.adapt(html)

        return text