from typing import List
import trafilatura
from application.adapters.web_adapter import WebAdapter


class TrafilaturaWebAdapter(WebAdapter):
    """
    Uses trafilatura to parse HTML's relevant content or None if any error occurred.
    TODO: it's useful to pass the URL too as there is a known bug in some cases (see https://github.com/adbar/trafilatura/issues/75)
    """
    @staticmethod
    def adapt(htmls: List[str]) -> List[str]:
        res = []
        for html in htmls:
            extracted = trafilatura.extract(
                html,
                include_links=False,
                include_images=False,
                include_tables=False,
                favor_precision=True
            )

            if extracted:
                res.append(extracted)

        return res

