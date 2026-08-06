import logging

def setup_logging() -> logging.Logger:
    logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("mydigitaltranslator")
    logger.setLevel(logging.DEBUG)
    return logger