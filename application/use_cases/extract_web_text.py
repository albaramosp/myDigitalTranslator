from typing import List
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

    def execute(self, urls: List[str]) -> List[str]:
        htmls = []
        for url in urls:
            html = self.downloader.download(url)
            if html:
                htmls.append(html)
        return self.extractor.adapt(htmls)
