import asyncio
import time

def task():
    x = 1
    for _ in range(10000):
        for _ in range(10000):
            x += 1
    return x

async def async_task():
    x = await asyncio.to_thread(task)
    return x

async def test():
    st = time.time()
    t = asyncio.create_task(async_task())
    res = await t
    dur = time.time() - st

    st = time.time()
    res2 = task()
    dur2 = time.time() - st
    assert res == res2 
    print(f"fake async takes {dur:.2f}, sync takes {dur2:.2f}")

def main():
    asyncio.run(test())

if __name__ == "__main__":
    main()
