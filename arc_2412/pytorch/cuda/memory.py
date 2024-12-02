"""
- torch.cuda.memory_allocated() returns the current GPU memory usage.
    - It is the total amount of memory that has been allocated by the allocator.
    - It accounts for the memory allocated on a particular device on a particular process.
    - Cross process memory allocation is not tracked.

(side note)
- Python encapsulates exceptions in Future objects which siliently suppresses the 
    exception if not handled properly.
    - Always call fut.result() for async executions!!!
- multiprocessing.Event cannot be passed to a process pool executor. 
    - Use Manager().Event() instead.
"""

import concurrent.futures
import multiprocessing
import time
from multiprocessing import Manager

import torch


def process(idx, size, event):
    torch.cuda.set_device(0)
    t = torch.rand(*size, device="cuda")
    mem = torch.cuda.memory_allocated() / (1024**2)
    print(f"Process {idx}: {mem:.2f} MB")
    while not event.is_set():
        time.sleep(1)


def main():
    multiprocessing.set_start_method("spawn")
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    mem = torch.cuda.memory_allocated() / (1024**2)
    print(f"Main: {mem:.2f} MB")

    N = 5
    print("Starting processes")
    with Manager() as manager:
        event = manager.Event()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futs = [executor.submit(process, i, (i, 1024, 1024), event) for i in range(N)]
            for _ in range(3):
                mem = torch.cuda.memory_allocated() / (1024**2)
                print(f"Main: {mem:.2f} MB")
                print("Sleeping for 3 seconds")
                time.sleep(3)
            event.set()
            for fut in futs:
                fut.result()

    print("All processes are done. Check memory again")
    mem = torch.cuda.memory_allocated() / (1024**2)
    print(f"Main: {mem:.2f} MB")


if __name__ == "__main__":
    main()
