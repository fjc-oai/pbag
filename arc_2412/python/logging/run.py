"""
- logger 
    1. hierarchy & propagation
    2. severity level
- handler
    1. stream
    2. file
    3. formatter
- __name__ is path_to_module_under_package/module_name
- print vs logging
"""

import logging
from config_logging import config_logging
from infra.trainer import fn as infra_fn
from ml.model import fn as ml_fn
from infra.trainer import get_name as infra_get_name
from ml.model import get_name as ml_get_name

logger = logging.getLogger(__name__)

def check_name():
    assert __name__ == "__main__"
    assert infra_get_name() == "infra.trainer"
    assert ml_get_name() == "ml.model"

def check_file_log():
    with open("infra.log", "r") as f:
        content = f.read()
        assert "a infra.trainer message" in content
        assert "a infra.trainer debug message" not in content
        assert "a infra.trainer print message" not in content
        assert "a ml.model message" not in content
        assert "Init logging" not in content


def main():
    check_name()
    logger.info("Init logging")
    config_logging()
    logger.info("Start main")
    infra_fn()
    ml_fn()
    logger.info("End main")

    check_file_log()

if __name__ == "__main__":
    main()


