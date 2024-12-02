import asyncio


async def producer(idx, q):
    for i in range(idx + 1):
        await q.put(f"{idx}-{i}")


async def consumer(idx, q):
    while True:
        item = await q.get()
        print(f"Consumer {idx} got {item}")
        q.task_done()


async def run():
    q = asyncio.Queue()
    N_PRODUCERS = 3
    N_CONSUMERS = 5
    producers = [asyncio.create_task(producer(i, q)) for i in range(N_PRODUCERS)]
    consumers = [asyncio.create_task(consumer(i, q)) for i in range(N_CONSUMERS)]
    await asyncio.gather(*producers)
    await q.join() # Implicitly awaits consumers!
    for c in consumers:
        c.cancel()


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
