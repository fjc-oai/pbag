import logging

logger = logging.getLogger(__name__)

def fn():
    logger.info("a infra.trainer message")
    logger.debug("a infra.trainer debug message")
    print("a infra.trainer print message")

def get_name():
    return __name__