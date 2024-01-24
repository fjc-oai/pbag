import backoff
import logging

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh_fmt = logging.Formatter("%(asctime)s-%(name)s: %(message)s", datefmt="%H:%M:%S")
sh.setFormatter(sh_fmt)
root_logger.addHandler(sh)

logger = logging.getLogger(__name__)


class UnreliableClient:
    def __init__(self):
        self.attempts = 0

    def get(self):
        logger.info("Attempt %d", self.attempts)
        if self.attempts < 2:
            self.attempts += 1
            raise IOError("Service unavailable. Try again later.")
        return "response"


def query_without_retry(client: UnreliableClient):
    return client.get()


@backoff.on_exception(lambda: backoff.expo(factor=10.0, base=1.5, max_value=30.0), IOError, max_tries=3)
def query(client: UnreliableClient):
    return client.get()


def main():
    client = UnreliableClient()
    result = query(client)
    logger.info(f"Result: {result}")


if __name__ == "__main__":
    main()
