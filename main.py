import logging
from application.use_cases.extract_web_text import ExtractWebText
from application.use_cases.adapt_web_text import AdaptWebText
from infrastructure.llm.llm_factory import LlmFactory
from infrastructure.requests_web_downloader import RequestsWebDownloader
from infrastructure.trafilatura_parser import TrafilaturaWebAdapter
from logging_config import setup_logging
import gradio


def analyze(user_input: str) -> str:
    error_msg = "An error occurred. Please contact the site administrator."

    try:
        text = ExtractWebText(
            downloader=RequestsWebDownloader(),
            adapter=TrafilaturaWebAdapter()
        ).execute(user_input)

        if not text:
            return error_msg

        return AdaptWebText(
            LlmFactory.create()
        ).execute(text)

    except Exception as err:
        logger = logging.getLogger("mydigitaltranslator")
        logger.error(err)
        return error_msg


if __name__ == '__main__':
    setup_logging()

    demo = gradio.Interface(
        fn=analyze,
        inputs="text",
        outputs="text",
        title="Introduce the URL to be analyzed"
    )

    demo.launch()



