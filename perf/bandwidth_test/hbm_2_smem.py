import argparse

import torch
import triton
import triton.language as tl


@triton.jit
def hbm_to_smem_kernel(src_ptr, dst_ptr, n_elements, iters: tl.constexpr, BLOCK_SIZE: tl.constexpr):
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


def benchmark_hbm_to_smem(
    n_elements=1024 * 1024 * 256,
    block_size=256,
    iters=100,
):
    src = torch.randn(n_elements, dtype=torch.float32, device="cuda")
    dst = torch.zeros_like(src)
    grid = ((n_elements + block_size - 1) // block_size,)

    # Warm up
    for _ in range(3):
        hbm_to_smem_kernel[grid](src, dst, n_elements, iters, BLOCK_SIZE=block_size)
    torch.cuda.synchronize()

    e1 = torch.cuda.Event(enable_timing=True)
    e2 = torch.cuda.Event(enable_timing=True)
    e1.record()
    hbm_to_smem_kernel[grid](src, dst, n_elements, iters, BLOCK_SIZE=block_size)
    e2.record()
    torch.cuda.synchronize()
    dur = e1.elapsed_time(e2) / 1e3  # in seconds

    total_bytes = n_elements * src.element_size() * iters
    gb_s = (total_bytes / 1e9) / dur
    print(f"Bandwidth B{block_size} I{iters}: {gb_s:.2f} GB/s in {dur:.4} s")


def benchmark_auto():
    N = 1024 * 1024 * 1024  # 4GB of data
    B = [64, 256, 1024, 4096]
    I = [1, 2, 5, 10, 20]
    for b in B:
        for i in I:
            benchmark_hbm_to_smem(n_elements=N, block_size=b, iters=i)


def main():
    argparser = argparse.ArgumentParser(description="HBM to SMEM bandwidth benchmark")
    argparser.add_argument("-N", type=int, default=1024 * 1024 * 256)  # 1GB of data
    argparser.add_argument("-B", type=int, default=256, help="Block size")
    argparser.add_argument("-I", type=int, default=3, help="Number of iterations")
    argparser.add_argument("--auto", action="store_true", help="Run with default settings")
    args = argparser.parse_args()

    if args.auto:
        benchmark_auto()
    else:
        benchmark_hbm_to_smem(n_elements=args.N, block_size=args.B, iters=args.I)


if __name__ == "__main__":
    main()
