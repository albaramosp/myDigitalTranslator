from abc import ABC, abstractmethod

class WebDownloader(ABC):
    @staticmethod
    @abstractmethod
    def download(url: str) -> str:
        pass