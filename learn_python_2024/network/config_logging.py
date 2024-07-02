import logging
import logging.config

TINY_ASYNCIO_TO_CONSOLE = False

def setup_logging():
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file_handler": {
                "class": "logging.FileHandler",
                "formatter": "standard",
                "filename": "a.log",
                "mode": "w",  # or 'a' for append
            },
        },
        "loggers": {
            "tiny_asyncio": {
                "level": "DEBUG",
                "handlers": ["file_handler"] + (["console"] if TINY_ASYNCIO_TO_CONSOLE else []),
                "propagate": False,
            },
            "": {
                "level": "INFO",
                "handlers": ["console", "file_handler"],
                "propagate": False,
            },
        },
    }

    logging.config.dictConfig(logging_config)
