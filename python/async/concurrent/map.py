"""
map(fn, iterable) is similar to [fn(x) for x in iterable].
Results are lazy evaluated.
"""

import time
import concurrent.futures

executor_fn = None


def get_executor():
    global executor_fn
    return executor_fn()


def set_executor(executor_fn_):
    global executor_fn
    executor_fn = executor_fn_


def io_task(x):
    time.sleep(1)
    return x * x


def test_io():
    print("Starting test_io")
    st = time.time()
    with get_executor() as executor:
        fut = executor.submit(io_task, 1)
    fut.result()
    dur = time.time() - st
    print(f"Process one task: {dur:.2f} seconds")

    st = time.time()
    N = 3
    with get_executor() as executor:
        results = list(executor.map(io_task, range(N)))
    assert results == [x * x for x in range(N)]
    dur2 = time.time() - st
    print(f"Process {N} tasks: {dur2:.2f} seconds")


def cpu_task(x):
    for _ in range(10000):
        for _ in range(10000):
            x = x * 1
    return x


def test_cpu():
    print("Starting test_cpu")
    st = time.time()
    with get_executor() as executor:
        fut = executor.submit(cpu_task, 1)
    fut.result()
    dur = time.time() - st
    print(f"Process one task: {dur:.2f} seconds")

    st = time.time()
    N = 3
    with get_executor() as executor:
        results = list(executor.map(cpu_task, range(N)))
    assert results == [x for x in range(N)]
    dur2 = time.time() - st
    print(f"Process {N} tasks: {dur2:.2f} seconds")


def main():
    print("Test ThreadPoolExecutor")
    set_executor(concurrent.futures.ThreadPoolExecutor)
    test_io()
    test_cpu()

    print("Test ProcessPoolExecutor")
    set_executor(concurrent.futures.ProcessPoolExecutor)
    test_io()
    test_cpu()


if __name__ == "__main__":
    main()
