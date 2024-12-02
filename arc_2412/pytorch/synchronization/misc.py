"""
Learn pytorch cuda synchronization primitives.

1. torch.cuda.Event wait() vs synchronize(): 

wait() enforces operator execution order, i.e. all future work submitted to the
stream only get executed after the completion of recorded event. synchronize()
blocks the current CPU thread so future work doesn't even get submitted before
the completion of the event.

2. python object lifetime:

The lifetime of an object is determined by the reference count of the object.
Deallocation is triggered immediately when the reference count drops to zero. No
delayed deallocation, unless cycle detection done by garbage collector.

3. tensor lifetime:

Same as python object lifetime. The underlying memory is usually managed by
extra layer, e.g. CachingAllocator, thus cudaFree() is not neccessarily always
called. No kernel launch is needed in either ways.

4. tensor lifetime in async ops:

Seemingly users are supposed to ensure the lifetime of the tensor expands until
the completion of the async ops. However, pytorch has a mechanism that
automatically ensures the tensor is not deallocated until the completion of op
on the corresponding stream. 


5. overall pytorch cuda concurrency:

a. execution order: 1) operations on the same stream execute in the same order
   as they are submitted to the stream. 2) use events to enforce execution order
   across streams.

b. lifetime management: 1) tensors are not deallocated until the completion of 
    the operation on the corresponding stream. 2) objects are deallocated
    immediately when the reference count drops to zero.
"""

import time

import torch


def _warmup_cuda():
    torch.cuda.synchronize()
    x = torch.randn(128, 128, device="cuda")
    y = torch.randn(128, 128, device="cuda")
    for _ in range(5):
        x = x @ x
        y = y @ y
    torch.cuda.synchronize()


def event_wait_vs_sync():
    _warmup_cuda()
    DIM = 4096
    N_ITER = 5
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_stack=True,
    ) as prof:
        x = torch.randn(DIM, DIM, device="cuda")
        y = torch.randn(DIM, DIM, device="cuda")
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        evt1 = torch.cuda.Event()
        evt2 = torch.cuda.Event()
        torch.cuda.synchronize()
        with torch.profiler.record_function("event_wait"):
            for _ in range(5):
                with torch.cuda.stream(s1):
                    evt2.wait()
                    x = x @ x
                    evt1.record()
                with torch.cuda.stream(s2):
                    evt1.wait()
                    y = y @ y
                    evt2.record()

        torch.cuda.synchronize()
        with torch.profiler.record_function("event_sync"):
            for _ in range(N_ITER):
                with torch.cuda.stream(s1):
                    evt2.synchronize()
                    x = x @ x
                    evt1.record()
                with torch.cuda.stream(s2):
                    evt1.synchronize()
                    y = y @ y
                    evt2.record()
    prof.export_chrome_trace("trace.json")


class Context:
    def __init__(self):
        self._stage_buffer = None


class SomeObject:
    def __init__(self, id):
        self.id = id
        print(f"SomeObject.__init__: {self.id}")

    def __del__(self):
        print(f"SomeObject.__del__: {self.id}")


def object_lifetime():
    ctx = Context()
    for idx in range(3):
        obj = SomeObject(idx)
        time.sleep(2)
        ctx._stage_buffer = obj
        time.sleep(2)


def _print_cuda_mem_usage():
    print(f"cuda memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f}MB", flush=True)
    print(f"cuda memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f}MB", flush=True)


def tensor_lifetime():
    context = Context()
    _print_cuda_mem_usage()
    for _ in range(10):
        x = torch.randn(1024**2, device="cuda")
        context._stage_buffer = x
        _print_cuda_mem_usage()


def _cuda_sleep(x):
    N_CYCLES = 1_000_000_000
    torch.cuda._sleep(x * N_CYCLES)


def tensor_lifetime_in_async_ops_pytorch():
    N_ITER = 10
    SIZE = 1024**2 * 128
    res = torch.zeros(N_ITER, device="cuda")
    context = Context()
    streams = [torch.cuda.Stream() for _ in range(N_ITER)]
    events = [torch.cuda.Event() for _ in range(N_ITER)]
    _print_cuda_mem_usage()
    x = torch.zeros(SIZE, device="cuda")
    for idx in range(N_ITER):
        with torch.cuda.stream(streams[idx]):
            x.copy_(torch.full((SIZE,), idx * 1.0, device="cuda"))
            context._stage_buffer = x
            t1 = torch.randn(1024*16, 1024*16, device="cuda")
            t2 = torch.randn(1024*16, 1024*16, device="cuda")
            t = t1 @ t2
            # _cuda_sleep(10)
            z = x.sum()
            res[idx] = z
            _print_cuda_mem_usage()
            events[idx].record()
    for idx in range(N_ITER):
        with torch.cuda.stream(streams[idx]):
            events[idx].wait()
            print(f"event {idx} done", flush=True)
    torch.cuda.synchronize()
    print("after sync", flush=True)
    _print_cuda_mem_usage()
    for idx in range(N_ITER):
        print(f"idx {idx}: expected={idx * SIZE}, actual={res[idx]}")
        # assert res[idx] == idx * SIZE


def main():
    # event_wait_vs_sync()
    # object_lifetime()
    # tensor_lifetime()
    tensor_lifetime_in_async_ops_pytorch()


if __name__ == "__main__":
    main()
