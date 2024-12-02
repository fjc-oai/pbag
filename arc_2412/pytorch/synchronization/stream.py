from concurrent.futures import ThreadPoolExecutor

import torch


def log(event, msg):
    event.synchronize()
    print(msg)

def task():
    a = torch.rand((1024 * 16, 1024 * 16), device='cuda')
    b = torch.rand((1024 * 16, 1024 * 16), device='cuda')
    c = torch.matmul(a, b)

def test_stream():
    torch.manual_seed(0)
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    ex = ThreadPoolExecutor(64)
    futs = []
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.cuda.stream(s1):
            for i in range(10):
                task()
                event = torch.cuda.Event()
                event.record(s1)
                futs.append(ex.submit(log, event, f"Stream 1 task {i} done"))
        # s2.wait_stream(s1)
        with torch.cuda.stream(s2):
            for i in range(10):
                task()
                event = torch.cuda.Event()
                event.record(s2)
                futs.append(ex.submit(log, event, f"Stream 2 task {i} done"))
            event = torch.cuda.Event()
        torch.cuda.synchronize()
    for fut in futs:
        fut.result()
    prof.export_chrome_trace("/tmp/trace.json")

test_stream()