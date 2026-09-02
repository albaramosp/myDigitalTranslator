from abc import ABC, abstractmethod
from typing import List


class WebAdapter(ABC):

    """
    Extracts meaningful data from an HTML content
    """
    @staticmethod
    @abstractmethod
    def adapt(htmls: List[str]) -> List[str]:
        pass