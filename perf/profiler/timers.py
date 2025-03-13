import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import torch
import triton
import triton.language as tl


def compute_task():
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)

    for i in range(1000):
        fibonacci(20)


def io_task():
    for _ in range(100):
        time.sleep(0.01)


def measure_wall_time(func):
    st = time.perf_counter()
    func()
    et = time.perf_counter()
    return et - st


def measure_cpu_time(func):
    st = time.thread_time()
    func()
    et = time.thread_time()
    return et - st


"""
Compute task:
    Wall time: 1.21 seconds
    CPU time: 1.20 seconds
IO task:
    Wall time: 1.01 seconds
    CPU time: 0.00 seconds
"""


def measure_time(func):
    start_wall_time = time.perf_counter()
    start_cpu_time = time.thread_time()
    func()
    end_wall_time = time.perf_counter()
    end_cpu_time = time.thread_time()
    return {"wall_time": end_wall_time - start_wall_time, "cpu_time": end_cpu_time - start_cpu_time}


def test_wall_and_cpu_time():
    print("Compute task:")
    print(f"    Wall time: {measure_wall_time(compute_task):.2f} seconds")
    print(f"    CPU time: {measure_cpu_time(compute_task):.2f} seconds")
    print("IO task:")
    print(f"    Wall time: {measure_wall_time(io_task):.2f} seconds")
    print(f"    CPU time: {measure_cpu_time(io_task):.2f} seconds")


"""
Task 0:
    Wall time: 4.79 seconds
    CPU time: 1.21 seconds
Task 1:
    Wall time: 3.84 seconds
    CPU time: 1.22 seconds
Task 2:
    Wall time: 2.82 seconds
    CPU time: 1.21 seconds
Task 3:
    Wall time: 3.70 seconds
    CPU time: 1.21 seconds
Task 4:
    Wall time: 3.12 seconds
    CPU time: 1.21 seconds
Task 5:
    Wall time: 2.44 seconds
    CPU time: 1.21 seconds
"""


def test_wall_and_cpu_time_with_threads():
    executor = ThreadPoolExecutor(max_workers=3)
    futures = [executor.submit(partial(measure_time, compute_task)) for _ in range(6)]
    for idx, future in enumerate(futures):
        result = future.result()
        print(f"Task {idx}:")
        print(f"    Wall time: {result['wall_time']:.2f} seconds")
        print(f"    CPU time: {result['cpu_time']:.2f} seconds")


@triton.jit(do_not_specialize=["duration"])
def _sleep(duration):
    start = tl.extra.cuda.globaltimer()
    while tl.extra.cuda.globaltimer() - start < duration:
        pass


def cuda_sleep(duration: float):
    _sleep[(1,)](int(duration * 1e9))


def cuda_task():
    print("cuda task start")
    x = torch.randn(1, device="cuda")
    cuda_sleep(1)
    # torch.cuda.synchronize()
    x.item()
    print("cuda task end")


def tick_task():
    for idx in range(40):
        print(f"Tick {idx}")
        time.sleep(0.1)


"""
cuda task start
Tick 0
...
Tick 22
cuda task end
Tick 23
...
Tick 39
Total time: 4.32 seconds
Cuda task:
    Wall time: 2.60 seconds
    CPU time: 1.66 seconds
CPU task:
    Wall time: 1.22 seconds
    CPU time: 1.20 seconds
"""


def test_cuda_sync_yield():
    st = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=3)
    cuda_fut = executor.submit(partial(measure_time, cuda_task))
    cpu_fut = executor.submit(partial(measure_time, compute_task))
    tick_fut = executor.submit(partial(measure_time, tick_task))
    cuda_res = cuda_fut.result()
    cpu_res = cpu_fut.result()
    tick_res = tick_fut.result()
    et = time.perf_counter()
    print(f"Total time: {et - st:.2f} seconds")
    print("Cuda task:")
    print(f"    Wall time: {cuda_res['wall_time']:.2f} seconds")
    print(f"    CPU time: {cuda_res['cpu_time']:.2f} seconds")
    print("CPU task:")
    print(f"    Wall time: {cpu_res['wall_time']:.2f} seconds")
    print(f"    CPU time: {cpu_res['cpu_time']:.2f} seconds")


if __name__ == "__main__":
    # test_wall_and_cpu_time()
    test_cuda_sync_yield()
