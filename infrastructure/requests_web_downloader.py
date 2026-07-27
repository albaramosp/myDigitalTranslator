import requests
from application.ports.web_downloader import WebDownloader


class RequestsWebDownloader(WebDownloader):
    @staticmethod
    def download(url: str) -> str:
        response = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;",
                "accept-language": "en-US,en;q=0.5",
            },
            timeout=10,
        )

        response.raise_for_status()

        return response.text