from abc import ABC, abstractmethod
from typing import Optional


class WebDownloader(ABC):
    @staticmethod
    @abstractmethod
    def download(url: str) -> Optional[str]:
        pass