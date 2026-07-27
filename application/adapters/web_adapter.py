from abc import ABC, abstractmethod
from typing import Optional


class WebAdapter(ABC):

    """
    Extracts meaningful data from an HTML content
    """
    @staticmethod
    @abstractmethod
    def adapt(html: str) -> Optional[str]:
        pass