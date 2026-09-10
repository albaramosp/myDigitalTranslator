import logging
from typing import List

from application.use_cases.adapt_multiple_web_texts import AdaptMultipleWebTexts
from application.use_cases.extract_web_text import ExtractWebText
from application.use_cases.semantic_search import SemanticSearch
from infrastructure.embeddings.gemini_embedding_client import GeminiEmbeddingClient
from infrastructure.llm.llm_factory import LlmFactory
from infrastructure.requests_web_downloader import RequestsWebDownloader
from infrastructure.trafilatura_parser import TrafilaturaWebAdapter
from logging_config import setup_logging
import gradio

def analyze_streaming(user_input: List[str]):
    error_msg = "An error occurred. Please contact the site administrator."

    try:
        texts = ExtractWebText(
            downloader=RequestsWebDownloader(),
            adapter=TrafilaturaWebAdapter()
        ).execute(user_input)

        if not texts:
            yield error_msg
            return

        accumulated = ""

        for chunk in AdaptMultipleWebTexts(
                LlmFactory.create()
        ).stream(texts):
            accumulated += chunk
            yield accumulated, []

        chunks = GeminiEmbeddingClient().embed(accumulated)

        yield accumulated, chunks

    except Exception as err:
        logger = logging.getLogger("mydigitaltranslator")
        logger.error(err)
        yield error_msg


def analyze(user_input: List[str]) -> str:
    error_msg = "An error occurred. Please contact the site administrator."

    try:
        texts = ExtractWebText(
            downloader=RequestsWebDownloader(),
            adapter=TrafilaturaWebAdapter()
        ).execute(user_input)

        if not texts:
            return error_msg

        return AdaptMultipleWebTexts(
            LlmFactory.create()
        ).execute(texts)

    except Exception as err:
        logger = logging.getLogger("mydigitaltranslator")
        logger.error(err)
        return error_msg


def search_relevant_content(user_input, chunks):
    c = SemanticSearch(GeminiEmbeddingClient(), LlmFactory.create())
    return c.semantic_search(user_input, chunks, 3)


if __name__ == '__main__':
    setup_logging()

    def add_url(current, session):
        session.append(current)
        return "", f"Total URLs: {len(session)}", session


    with gradio.Blocks() as demo:
        urls_state = gradio.State([])
        with gradio.Row():
            url_input = gradio.Textbox(label="Enter a URL")
            add_btn = gradio.Button("Save URL")

        status = gradio.Label("Total URLs = 0")

        with gradio.Accordion("Final Summary", open=False):
            report_btn = gradio.Button("Generate Summary", variant="primary")
            report_output = gradio.Markdown()

        add_btn.click(
            fn=add_url,
            inputs=[url_input, urls_state],
            outputs=[url_input, status, urls_state]
        )

        semantic_index_state = gradio.State([])

        accumulated = report_btn.click(
            fn=analyze_streaming,
            inputs=urls_state,
            outputs=[report_output, semantic_index_state]
        )

        with gradio.Accordion("Semantic Search"):
            search_box = gradio.Textbox(
                label="Ask about the summary"
            )
            search_btn = gradio.Button("Search")
            search_btn.click(
                fn=search_relevant_content,
                inputs=[search_box, semantic_index_state],
                outputs=gradio.Markdown()
            )

    demo.launch()



