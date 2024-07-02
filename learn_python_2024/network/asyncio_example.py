"""
- [x] basic async task execution: define with async def, and run with await
- [x] use asyncio.run() to drive the event loop
- [x] support async.sleep()
- [x] support socket/selector based async IO
- [x] schedule periodical task running on background
- [x] concurrent execution through asyncio.gather()
- [ ] proper error handling
- [ ] unit tests

Usage:
1. Demo execution flow, e.g. await, sleep, gather, periodical task

    $ python asyncio_example.py flow
    2024-07-01 16:38:03,759 - INFO - asyncio_example.py:81 - flow_demo starts
    2024-07-01 16:38:03,759 - INFO - asyncio_example.py:74 - task 1 start: sleep 5 seconds
    2024-07-01 16:38:03,759 - INFO - asyncio_example.py:74 - task 2 start: sleep 2 seconds
    2024-07-01 16:38:03,760 - INFO - asyncio_example.py:74 - task 3 start: sleep 10 seconds
    2024-07-01 16:38:05,763 - INFO - asyncio_example.py:76 - task 2 end
    2024-07-01 16:38:06,762 - INFO - asyncio_example.py:70 - health check: everything is fine
    2024-07-01 16:38:08,762 - INFO - asyncio_example.py:76 - task 1 end
    2024-07-01 16:38:09,764 - INFO - asyncio_example.py:70 - health check: everything is fine
    2024-07-01 16:38:12,767 - INFO - asyncio_example.py:70 - health check: everything is fine
    2024-07-01 16:38:13,762 - INFO - asyncio_example.py:76 - task 3 end
    2024-07-01 16:38:13,763 - INFO - asyncio_example.py:89 - All tasks done: 2, 1, 3
    2024-07-01 16:38:13,770 - INFO - asyncio_example.py:90 - flow_demo takes 10.003628015518188 seconds
    2024-07-01 16:38:13,770 - INFO - asyncio_example.py:112 - All done

2. Demo async IO, e.g. fetch data, process data, from httpbin.org

    $ python asyncio_example.py io
    2024-07-01 16:41:29,810 - INFO - asyncio_example.py:75 - async_io_demo starts
    2024-07-01 16:41:29,810 - INFO - asyncio_example.py:52 - Fetching data from httpbin.org:80 ...
    2024-07-01 16:41:29,991 - INFO - asyncio_example.py:79 - Fetch data takes 0.18 seconds
    2024-07-01 16:41:29,991 - INFO - asyncio_example.py:70 - Processing data...
    2024-07-01 16:41:29,991 - INFO - asyncio_example.py:82 - Process data takes 0.00 seconds
    2024-07-01 16:41:29,991 - INFO - asyncio_example.py:83 - Result: {'length': 425, 'n_words': 37}
    2024-07-01 16:41:29,992 - INFO - asyncio_example.py:130 - All done

3. Demo async IO from local socket server

    > python selector_example.py server
    Server is listening on localhost:12345
    Accepted connection from ('127.0.0.1', 53567)
    Partial message received: GET /get
    Waiting for more data...
    Partial message received:  HTTP/1.
    Waiting for more data...
    Full message received: GET /get HTTP/1.
    Now sending response: GET /GET HTTP/1.
    Sent 16 bytes: GET /GET HTTP/1.
    Data sent successfully. Now unregistering the write event.

    > python asyncio_example.py io --local
    2024-07-01 16:43:51,869 - INFO - asyncio_example.py:85 - async_io_demo starts
    2024-07-01 16:43:51,869 - INFO - asyncio_example.py:62 - Fetching data from localhost:12345 ...
    2024-07-01 16:43:51,871 - INFO - asyncio_example.py:89 - Fetch data takes 0.00 seconds
    2024-07-01 16:43:51,871 - INFO - asyncio_example.py:80 - Processing data...
    2024-07-01 16:43:51,871 - INFO - asyncio_example.py:92 - Process data takes 0.00 seconds
    2024-07-01 16:43:51,871 - INFO - asyncio_example.py:93 - Result: {'length': 16, 'n_words': 3}
"""

USE_TINY_ASYNCIO = True
if USE_TINY_ASYNCIO:
    import tiny_asyncio as asyncio
else:
    import asyncio

import argparse
import logging
import time

from config_logging import setup_logging

logger = logging.getLogger(__name__)


async def fetch_data(host, port):
    logger.info(f"Fetching data from {host}:{port} ...")
    request = "GET /get HTTP/1.1\r\nHost: httpbin.org\r\nConnection: close\r\n\r\n"
    if USE_TINY_ASYNCIO:
        sock = await asyncio.create_connection(host, port)
        n = await asyncio.sock_sendall(sock, request.encode("utf-8"))
        response = await asyncio.sock_recv(sock)
        return response
    else:
        reader, writer = await asyncio.open_connection(host, port)
        writer.write(request.encode("utf-8"))
        await writer.drain()
        response = await reader.read(4096)
        writer.close()
        await writer.wait_closed()
        return response.decode("utf-8")


async def process_data(data: str):
    logger.info("Processing data...")
    return {"length": len(data), "n_words": len(data.split())}


async def async_io_demo(host, port):
    logger.info("async_io_demo starts")
    cur = time.time()
    data = await fetch_data(host, port)
    after_fetch = time.time()
    logger.info(f"Fetch data takes {after_fetch - cur:.2f} seconds")
    result = await process_data(data)
    after_process = time.time()
    logger.info(f"Process data takes {after_process - after_fetch:.2f} seconds")
    logger.info(f"Result: {result}")
    return


def health_check():
    logger.info("health check: everything is fine")


async def task(id, dur):
    logger.info(f"task {id} start: sleep {dur} seconds")
    await asyncio.sleep(dur)
    logger.info(f"task {id} end")
    return id


async def flow_demo():
    logger.info("flow_demo starts")
    asyncio.create_periodical_task(health_check, 3)
    st = time.time()
    res = await asyncio.gather(
        task(1, 5),
        task(2, 2),
        task(3, 10),
    )
    logger.info("All tasks done: " + ", ".join(map(str, res)))
    logger.info(f"flow_demo takes {time.time() - st} seconds")


if __name__ == "__main__":
    setup_logging()

    argparser = argparse.ArgumentParser()
    argparser.add_argument("mode", choices=["io", "flow"], default="io")
    argparser.add_argument("--local", action="store_true")
    args = argparser.parse_args()

    if args.mode == "io":
        if args.local:
            host = "localhost"
            port = 12345
        else:
            host = "httpbin.org"
            port = 80
        asyncio.run(async_io_demo(host, port))
    elif args.mode == "flow":
        asyncio.run(flow_demo())
