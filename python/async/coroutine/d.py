import asyncio

async def task(x):
    return x + 1

async def main():
    t = asyncio.create_task(task(1))
    at = await t
    print(f"{at=}")

if __name__ == "__main__":
    asyncio.run(main())