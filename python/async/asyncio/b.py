import asyncio

async def task(x):
    await asyncio.sleep(3)
    return x + 1

async def main():
    t = task(1)
    print(f"{type(t)=}") # <class 'coroutine'>

    at = await t
    print(f"{type(at)=}") # int

if __name__ == "__main__":
    asyncio.run(main())