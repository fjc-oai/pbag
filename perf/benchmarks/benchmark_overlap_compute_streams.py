"""
Benchmark overlap compute streams.

1. Overlap two streams with matmul
    python benchmark_overlap_compute_streams.py --op matmul
    # Time taken with 1 stream: 3332.122802734375 ms
    # Time taken with 2 streams interleaved: 3315.3896484375 ms

2. Overlap two streams with add
    python benchmark_overlap_compute_streams.py --op add
    # Time taken with 1 stream: 14.314240455627441 ms
    # Time taken with 2 streams interleaved: 14.241312026977539 ms

3. Overlap two streams with sleep
    python benchmark_overlap_compute_streams.py --op sleep
    # Time taken with 1 stream: 2000.19677734375 ms
    # Time taken with 2 streams interleaved: 1000.2637939453125 ms

Run with torch profiler:
    python benchmark_overlap_compute_streams.py --profiler=torch

Run with nsys profiler:
    nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --cudabacktrace=true -x true -o my_profile python benchmark_overlap_comp
ute_streams.py  --profiler=nvtx --op=matmul

    With SM utilizations: --gpu-metrics-devices=all
"""

import argparse
from contextlib import nullcontext
from typing import Callable, Literal

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["duration"])
def _sleep(duration):
    start = tl.extra.cuda.globaltimer()
    while tl.extra.cuda.globaltimer() - start < duration:
        pass


def cuda_sleep(*args, **kwargs):
    _sleep[(1,)](0.1 * 1e9)


def task(
    xs: list[torch.Tensor],
    ops: list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    streams: list[torch.cuda.Stream],
    tag: str,
):
    with torch.profiler.record_function(f"{tag}"), torch.cuda.nvtx.range(f"{tag}"):
        for x_idx, x in enumerate(xs):
            stream = streams[x_idx % len(streams)]
            out = x
            with torch.cuda.stream(stream):
                for op in ops:
                    out = op(out, x)
        return out


def run_task(
    n: int,
    d: int,
    op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_streams: int,
):
    torch.manual_seed(7)
    tag = f"task_n{n}_d{d}_s{num_streams}"
    xs = [torch.randn(d, d, device="cuda") for _ in range(2)]
    ops = [op] * n
    streams = [torch.cuda.Stream() for _ in range(num_streams)]

    task(xs, ops, streams, f"{tag}-warmup")
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    task(xs, ops, streams, f"{tag}-benchmark")
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event)


def benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--d", type=int, default=1024 * 16)
    parser.add_argument("--op", type=str, default="matmul")
    parser.add_argument("--profiler", type=str, default="")
    args = parser.parse_args()

    N = args.n
    D = args.d
    assert args.op in ["matmul", "add", "sleep"]
    if args.op == "matmul":
        op = torch.matmul
    elif args.op == "add":
        op = torch.add
    elif args.op == "sleep":
        op = cuda_sleep
    else:
        raise ValueError(f"Invalid operation: {args.op}")

    assert args.profiler in ["nvtx", "torch", ""]
    if args.profiler == "torch":
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
        )
        prof.start()

    if args.profiler == "nvtx":
        torch.cuda.cudart().cudaProfilerStart()

    t1 = run_task(N, D, op, 1)
    print(f"Time taken with 1 stream: {t1} ms")

    t2 = run_task(N, D, op, 2)
    print(f"Time taken with 2 streams interleaved: {t2} ms")

    if args.profiler == "nvtx":
        torch.cuda.cudart().cudaProfilerStop()

    if args.profiler == "torch":
        prof.stop()
        prof.export_chrome_trace("trace.json")
        print("Trace exported to trace.json")


if __name__ == "__main__":
    benchmark()
