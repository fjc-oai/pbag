import asyncio
import time


async def say_after(delay, message):
    print(message + " starts")
    await asyncio.sleep(delay)
    print(message + " ends")

async def main():
    task1 = asyncio.create_task(say_after(2, 'hello'))
    task2 = asyncio.create_task(say_after(2, 'world'))

    print("Tasks started")
    now = time.time()
    await task1
    await task2
    print(f"Tasks finished after {time.time() - now}")

asyncio.run(main())