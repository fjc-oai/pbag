"""
Used for maintain the context of the request in async and sync code.

Support both asyncio task and concurrent thread, whereas threading.Local is only for thread.
"""
import asyncio
from contextvars import ContextVar
import time
from concurrent.futures import ThreadPoolExecutor

request_id = ContextVar("request_id")


async def process_request(id):
    token = request_id.set(id)
    await asyncio.sleep(1)
    x = request_id.get()
    await asyncio.sleep(1)
    y = request_id.get()
    await asyncio.sleep(1)
    print(f"Processing request {x} and {y} for {id}")
    assert x == y
    assert x == id
    request_id.reset(token)


async def use_asyncio():
    tasks = [process_request(i) for i in range(10)]
    await asyncio.gather(*tasks)


def async_test():
    asyncio.run(use_asyncio())


def process_request(id):
    token = request_id.set(id)
    time.sleep(1)
    x = request_id.get()
    time.sleep(1)
    y = request_id.get()
    time.sleep(1)
    print(f"Processing request {x} and {y} for {id}")
    assert x == y
    assert x == id
    request_id.reset(token)


def use_executor():
    with ThreadPoolExecutor(max_workers=10) as executor:
        tasks = [executor.submit(process_request, i) for i in range(10)]
        for task in tasks:
            task.result()


def main():
    # async_test()
    use_executor()


if __name__ == "__main__":
    main()
