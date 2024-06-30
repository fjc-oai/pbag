USE_TINY_ASYNCIO = True
if USE_TINY_ASYNCIO:
    import tiny_asyncio as asyncio
else:
    import asyncio

import logging
import time

from config_logging import setup_logging
from tqdm import tqdm

"""
- [x] basic sequence of async functions, defined with async def, and run with await
- [x] use asyncio.run() to run the main function
- [x] support sleep
- [x] schedule a task that runs in x seconds later
- schedule the task running on background periodically
- async fetch data
- concurrently run multiple tasks
"""

logger = logging.getLogger(__name__)


async def fetch_data():
    logger.info("start fetching data")
    for _ in tqdm(range(5)):
        await asyncio.sleep(1)
    return "python is the best programming language because"


async def process_data(data: str):
    logger.info("processing data...")
    l = len(data)
    return {"length": l}


async def main():
    cur = time.time()
    data = await fetch_data()
    after_fetch = time.time()
    logger.info(f"fetch data takes {after_fetch - cur:.2f} seconds")
    result = await process_data(data)
    after_process = time.time()
    logger.info(f"process data takes {after_process - after_fetch:.2f} seconds")
    logger.info(f"result: {result}")
    return 0


if __name__ == "__main__":
    setup_logging()
    asyncio.run(main())
