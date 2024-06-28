USE_TINY_ASYNCIO = True
if USE_TINY_ASYNCIO:
    import tiny_asyncio as asyncio
else:
    import asyncio

from tqdm import tqdm

"""
- [x] basic sequence of async functions, defined with async def, and run with await
- [x] use asyncio.run() to run the main function
- support sleep
- schedule a task that runs in x seconds later
- schedule the task running on background periodically
- async fetch data
- concurrently run multiple tasks
"""


async def fetch_data():
    print("start fetching data")
    for _ in tqdm(range(3)):
        await asyncio.sleep(1)
        print("fetching...")
    return "python is the best programming language because"


async def process_data(data: str):
    print("processing data...")
    l = len(data)
    return {"length": l}


async def main():
    data = await fetch_data()
    result = await process_data(data)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
