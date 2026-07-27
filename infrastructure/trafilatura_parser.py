from typing import Optional
import trafilatura
from application.adapters.web_adapter import WebAdapter


class TrafilaturaWebAdapter(WebAdapter):
    """
    Uses trafilatura to parse HTML's relevant content or None if any error occurred.
    TODO: it's useful to pass the URL too as there is a known bug in some cases (see https://github.com/adbar/trafilatura/issues/75)
    """
    @staticmethod
    def adapt(html: str) -> Optional[str]:
        return trafilatura.extract(
            html,
            include_links=False,
            include_images=False,
            include_tables=False,
            favor_precision=True
        )