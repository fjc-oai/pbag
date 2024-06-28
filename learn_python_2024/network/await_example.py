import asyncio


async def bar():
    print("bar running")
    breakpoint()
    return 7

async def foo():
    print("foo running")
    x = await bar()
    print(f"foo calling bar and got {x=}")
    return 10

async def main():
    x = await foo()
    print(f"{x=}")

# asyncio.run(main())
x = foo()
breakpoint()
x.send(None)