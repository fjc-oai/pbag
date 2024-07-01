USE_TINY_ASYNCIO = True
if USE_TINY_ASYNCIO:
    import tiny_asyncio as asyncio
else:
    import asyncio

import itertools
import logging
import time

from config_logging import setup_logging
from tqdm import tqdm

"""
- [x] basic sequence of async functions, defined with async def, and run with await
- [x] use asyncio.run() to run the main function
- [x] support sleep
- [x] schedule a task that runs in x seconds later
- [x] schedule the task running on background periodically
- [x] async fetch data through socket
- [x] concurrently run multiple tasks
"""

logger = logging.getLogger(__name__)


async def fetch_data_dummy(host, port):
    logger.info(f"start fetching data from {host}:{port}...")
    for _ in tqdm(range(5)):
        await asyncio.sleep(1)
    return "python is the best programming language because"


async def fetch_data_http(host, port):
    if USE_TINY_ASYNCIO:
        sock = await asyncio.create_connection(host, port)
        # request = "niuleBile\r\n\r\n" # use \r\n\r\n as delimiter to end the request
        request = "GET /get HTTP/1.1\r\nHost: httpbin.org\r\nConnection: close\r\n\r\n"
        n = await asyncio.sock_sendall(sock, request.encode("utf-8"))
        logger.info(f"sent {n} bytes")
        response = await asyncio.sock_recv(sock)
        logger.info(f"received {response}")
        return response
    else:
        reader, writer = await asyncio.open_connection(host, port)
        request = f"GET / HTTP/1.1\r\nHost: {host}\r\n\r\n"
        writer.write(request.encode("utf-8"))
        await writer.drain()
        response = await reader.read(4096)
        writer.close()
        await writer.wait_closed()
        return response.decode("utf-8")


async def process_data(data: str):
    logger.info("processing data...")
    l = len(data)
    return {"length": l, "first_50_chars": data[:50]}


def health_check():
    logger.info("health check: everything is fine")

async def main():
    if USE_TINY_ASYNCIO:
        asyncio.create_periodical_task(health_check, 3)

    cur = time.time()
    host = "httpbin.org"
    port = 80
    # host = "localhost"
    # port = 12345
    data = await fetch_data_http(host, port)
    after_fetch = time.time()
    logger.info(f"fetch data takes {after_fetch - cur:.2f} seconds")
    result = await process_data(data)
    after_process = time.time()
    logger.info(f"process data takes {after_process - after_fetch:.2f} seconds")
    logger.info(f"result: {result}")
    for i in range(100):
        logger.info(f"main loop idling: {i}")
        await asyncio.sleep(1)
    return 0


# async def main():
#     for i in range(10):
#         logger.info(f"main loop idling: {i}")
#         await asyncio.sleep(1)


async def fn(dur):
    logger.info(f"fn start: sleep {dur} seconds")
    await asyncio.sleep(dur)
    logger.info("fn end: slept {dur} seconds")
    return dur

async def main():
    st = time.time()
    logger.info("main start")
    res = await asyncio.gather(
        fn(3),
        fn(5),
        fn(2),
    )
    logger.info("main end")
    logger.info(f"res: {res}")
    logger.info(f"main takes {time.time() - st} seconds")


if __name__ == "__main__":
    setup_logging()
    asyncio.run(main())
