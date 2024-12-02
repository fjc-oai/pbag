import logging

logger = logging.getLogger(__name__)

def fn():
    logger.info("a ml.model message")

def get_name():
    return __name__