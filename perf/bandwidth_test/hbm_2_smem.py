"""
On H100, tested mem bandwidth is 2597 GB/s, pretty close to reported 3TB/s.

> $python hbm_2_smem.py -I=10
Iteration 0 took 0.826 ms
Iteration 1 took 0.783 ms
Iteration 2 took 0.766 ms
Iteration 3 took 0.767 ms
Iteration 4 took 0.764 ms
Iteration 5 took 0.770 ms
Iteration 6 took 0.774 ms
Iteration 7 took 0.760 ms
Iteration 8 took 0.760 ms
Iteration 9 took 0.760 ms
Bandwidth 262144x1024x10: 20.0 GB in 7.731 s -> 2587.006 GB/s
"""

import argparse

import torch
import triton
import triton.language as tl


@triton.jit
def hbm_to_smem_kernel(
    src_ptr, dst_ptr, n_elements, iters: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    smem = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(iters):
        # Iterate over disjoint sets of elements to avoid the impact of cache.
        # Not really sure if this is correctly implemented. With different iters
        # the results differ a lot...
        pid = tl.program_id(0) + i * 1024
        start_idx = pid * BLOCK_SIZE
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(src_ptr + offsets, mask=mask)
        smem += x
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(dst_ptr + offsets, smem, mask=mask)


@triton.jit
def _copy_kernel(
    Src,
    Dst,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (off_m[:, None] < M) & (off_n[None, :] < N)
    off = off_m[:, None] * N + off_n[None, :]
    x = tl.load(Src + off, mask=mask)
    # is tl.debug_barrier() needed here? probably not since address are different
    tl.store(Dst + off, x, mask=mask)


def copy(src, dst):
    assert src.dtype == dst.dtype
    assert src.shape == dst.shape
    assert src.is_contiguous()
    assert dst.is_contiguous()
    M, N = src.shape
    BLOCK_M = 16
    BLOCK_N = 512
    M_BLOCKS = triton.cdiv(M, BLOCK_M)
    N_BLOCKS = triton.cdiv(N, BLOCK_N)
    grid = (M_BLOCKS, N_BLOCKS)
    _copy_kernel[grid](src, dst, M, N, BLOCK_M, BLOCK_N)


def benchmark_mem_copy(
    m: int = 256 * 1024,
    n: int = 1024,
    iters: int = 100,
    check_correctness: bool = False,
):
    src = torch.randn(m, n, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)

    # Warm up
    copy(src, dst)
    torch.cuda.synchronize()

    dur = 0
    for i in range(iters):
        src = torch.randn(m, n, dtype=torch.float32, device="cuda")
        dst = torch.empty_like(src)
        torch.cuda.synchronize()

        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        event_start.record()
        copy(src, dst)
        event_end.record()
        torch.cuda.synchronize()
        print(f"Iteration {i} took {event_start.elapsed_time(event_end):.3f} ms")
        dur += event_start.elapsed_time(event_end)
        if check_correctness:
            assert torch.equal(src, dst)
    total_bytes = m * n * src.element_size() * iters * 2  # 2x for read and write
    gb_s = (total_bytes / (1024**3)) / dur * 1000  # convert to s
    print(
        f"Bandwidth {m}x{n}x{iters}: {total_bytes / (1024 ** 3)} GB in {dur:.4} s -> {gb_s:.3f} GB/s"
    )


def main():
    argparser = argparse.ArgumentParser(description="HBM to SMEM bandwidth benchmark")
    argparser.add_argument("-M", type=int, default=256 * 1024)  # 1GB of data
    argparser.add_argument("-N", type=int, default=1024)
    argparser.add_argument("-I", type=int, default=3, help="Number of iterations")
    argparser.add_argument("-C", type=bool, default=False, help="Check correctness")
    args = argparser.parse_args()

    benchmark_mem_copy(m=args.M, n=args.N, iters=args.I)


if __name__ == "__main__":
    main()
