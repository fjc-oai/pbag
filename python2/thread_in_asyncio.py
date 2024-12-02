import asyncio
from concurrent.futures import Future, ThreadPoolExecutor

executor = None


def get_executor():
    global executor
    if executor is None:
        executor = ThreadPoolExecutor(max_workers=4)
    return executor


def process(xs):
    return [x * x for x in xs]


def process_async(xs):
    print(f"process_asyncs: {xs}")
    return get_executor().submit(process, xs)


class AsyncSquare:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.process_task = None

    async def compute_square(self, x):
        if self.process_task is None:
            self.process_task = asyncio.create_task(self.process_queue())

        future = asyncio.get_event_loop().create_future()
        await self.queue.put((x, future))
        return await future

    async def process_queue(self):
        while True:
            pending = []
            while len(pending) < 4:
                x, future = await self.queue.get()
                pending.append((x, future))
            print(f"process_queue full. now processing")
            xs = [x for x, _ in pending]
            futures = [future for _, future in pending]
            try:
                res = await asyncio.get_event_loop().run_in_executor(self.executor, process_async, xs)
                ############################################################
                #
                # Prolbem:
                # This is the key part. res is in type of concurrent.futures.Future. It's not iterable.
                # 1. zip(res) will raise exception.
                # 2. since it's a background task, the exception won't be automatically propagated to the main thread.
                # 
                # Fix:
                # 1. wrap the res with asyncio.wrap_future to convert it to asyncio.Future, and await it.
                # 2. try-except block to catch the exception and set it to the future.  
                # 
                ############################################################
                res = await asyncio.wrap_future(res)
                for future, r in zip(futures, res):
                    future.set_result(r)
            except Exception as e:
                print(f"process_queue failed: {e}")
                for future in futures:
                    future.set_exception(e)

async def test():
    foo = AsyncSquare()
    tasks = [foo.compute_square(x) for x in range(4)]
    res = await asyncio.gather(*tasks)
    return res

async def test2():
    res = Future()
    res.set_result([1,2,3])
    futures = [asyncio.get_event_loop().create_future() for _ in range(3)]
    for fut, r in zip(futures, res):
        fut.set_result(r)

def main():
    res = asyncio.run(test())
    print(res)

    # asyncio.run(test2())

if __name__ == "__main__":
    main()
