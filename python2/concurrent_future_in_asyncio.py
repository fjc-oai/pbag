import asyncio
import concurrent.futures
import time


def thread_task(n) -> int:
    for i in range(n):
        print(f"thread_task at step {i}")
        time.sleep(1)
    return n

async def thread_task_wrapper(n) -> int:
    ex = concurrent.futures.ThreadPoolExecutor()
    return await asyncio.get_event_loop().run_in_executor(ex, thread_task, n)

async def asyncio_task(n) -> int:
    for i in range(n):
        print(f"asyncio_task at step {i}")
        await asyncio.sleep(1)
    return n


async def main() -> None:
    t1 = thread_task_wrapper(5)
    t2 = asyncio_task(5)
    res = await asyncio.gather(t1, t2)
    assert res == [5, 5]

if __name__ == '__main__':
    asyncio.run(main())