class Bar:
    def __init__(self):
        pass

    def __await__(self):
        for i in range(10):
            yield i
        return 7
    
async def fn1():
    print("fn1 running")
    x = await Bar()
    print(f"fn1 calling bar and got {x=}")
    return x

async def fn2():
    print("fn2 running")
    x = await fn1()
    print(f"fn2 calling fn1 and got {x=}")
    return x

async def fn3():
    print("fn3 running")
    x = await fn2()
    print(f"fn3 calling fn2 and got {x=}")
    return x

async def main():
    x = await fn3()
    print(f"main calling fn3 and got {x=}")

# asyncio.run(main())
x = main()

while True:
    try:
        print("----------> x.send() ------------>")
        y = x.send(None)
        print(f"got {y=}")
    except StopIteration as e:
        print("StopIteration received")
        break

print("All done")