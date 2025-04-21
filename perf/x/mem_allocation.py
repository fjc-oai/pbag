import torch
import tracemalloc


def test_tracemalloc():
    t1 = torch.randn(128, 1024, 1024)
    t2 = torch.randn(128, 1024, 1024, device="cuda")

    tracemalloc.start()

    s1 = tracemalloc.take_snapshot()

    t3 = torch.randn(128, 1024, 1024)
    t4 = torch.randn(128, 1024, 1024, device="cuda")

    s2 = tracemalloc.take_snapshot()

    print("-" * 100)
    print("s1")
    for record in s1.statistics("lineno"):
        print(record)

    print("-" * 100)
    print("s2")
    for record in s2.statistics("lineno"):
        print(record)

    print("-" * 100)
    print("diff")
    for record in s2.compare_to(s1, "lineno"):
        print(record)



def test_lazy_allocation():
    import time
    import numpy
    import os

    print(f"pid: {os.getpid()}")
    for i in range(20):
        print(f"sleeping {i}")
        time.sleep(1)
    print("creating a big array")
    big_array = numpy.empty(1_000_000_000)
    time.sleep(5)
    print("writing to the big array")
    big_array[:] = 42.0
    time.sleep(5)


test_lazy_allocation()
