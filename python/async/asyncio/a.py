import logging
import asyncio
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def task(x):
    await asyncio.sleep(3)
    return x + 1


def main():
    st = datetime.now()
    # asyncio.run() wraps a coroutine into a task 
    # and runs it. It blocks untils coroutine completes. 
    # So that we don't need to propagate async keyword 
    # to the top level.
    res = asyncio.run(task(1))
    assert res == 2
    dur = datetime.now() - st
    assert dur.seconds == 3


if __name__ == "__main__":
    main()
