import triton
import triton.language as tl
import torch
from tabulate import tabulate


@triton.jit
def add_kernel(
    A,
    B,
    Out,
    M,
    N,
    stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # num_warps: NO need to define num_warps here. Likely defined inside triton.jit already.
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (off_m[:, None] < M) & (off_n[None, :] < N)
    off = off_m[:, None] * stride + off_n[None, :]
    a = tl.load(A + off, mask=mask, other=0)
    b = tl.load(B + off, mask=mask, other=0)
    c = a + b
    tl.store(Out + off, c, mask=mask)


class Add(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, BLOCK_M, BLOCK_N, num_warps):
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        assert a.is_contiguous()
        assert b.is_contiguous()
        out = torch.empty_like(a)
        M, N = a.shape
        M_BLOCKS = triton.cdiv(M, BLOCK_M)
        N_BLOCKS = triton.cdiv(N, BLOCK_N)
        grid = (M_BLOCKS, N_BLOCKS)
        add_kernel[grid](
            a, b, out, M, N, a.stride(0), BLOCK_M, BLOCK_N, num_warps=num_warps
        )
        return out


def add(a, b, BLOCK_M=4, BLOCK_N=128, num_warps=1):
    return Add.apply(a, b, BLOCK_M, BLOCK_N, num_warps)


def test_correctness():
    M, N = 1024, 32
    a = torch.randn(M, N, device="cuda")
    b = torch.randn(M, N, device="cuda")
    out = add(a, b)
    ref = a + b
    assert torch.allclose(out, ref)
    print("Correctness test passed")


def run(M, N, BLOCK_M, BLOCK_N, num_warps):
    a = torch.randn(M, N, device="cuda")
    b = torch.randn(M, N, device="cuda")
    out = add(a, b, BLOCK_M, BLOCK_N, num_warps)
    event_start = torch.cuda.Event(enable_timing=True)
    event_end = torch.cuda.Event(enable_timing=True)
    event_start.record()
    N_ITER = 10
    for _ in range(N_ITER):
        out = add(a, b, BLOCK_M, BLOCK_N, num_warps)
    event_end.record()
    torch.cuda.synchronize()
    dur = event_start.elapsed_time(event_end) / N_ITER
    print(
        f"M: {M}, N: {N}, BLOCK_M: {BLOCK_M}, BLOCK_N: {BLOCK_N}, num_warps: {num_warps}, time: {dur:.3f}ms"
    )
    return dur


def run_torch(M, N):
    a = torch.randn(M, N, device="cuda")
    b = torch.randn(M, N, device="cuda")
    out = a + b
    event_start = torch.cuda.Event(enable_timing=True)
    event_end = torch.cuda.Event(enable_timing=True)
    event_start.record()
    N_ITER = 10
    for _ in range(N_ITER):
        out = a + b
    event_end.record()
    torch.cuda.synchronize()
    dur = event_start.elapsed_time(event_end) / N_ITER
    print(f"M: {M}, N: {N}, time: {dur:.3f}ms")
    return dur


################################################################################
#
#   Benchmark the performance of 1D and 2D grid layouts.
#            M    N    BLOCK_M    BLOCK_N    num_warps  time      time_torch
#      -------  ---  ---------  ---------  -----------  --------  ------------
#      4194304  480          1        256            1  8.093ms   7.777ms
#      4194304  480          2        256            1  8.117ms   7.777ms
#      4194304  480          4        256            1  8.148ms   7.776ms
#      4194304  480          8        256            1  8.130ms   7.776ms
#      4194304  480         64        256            1  25.525ms  7.776ms
#      4194304  480        256        256            1  28.131ms  7.776ms
#      4194304  480          1        512            2  7.814ms   7.776ms
#
#   - Intuitively, it might feel like 2D grid layout is faster due
#       to more parallelism. But it's not really the case at all!
#   - As long as the block size is chosen properly, 1D grid layout
#       can be as fast as 2D grid layout.
#   - Either paralleling on M or N dimensions is equally good.
#
################################################################################
def benchmark_1d_vs_2d_grid():
    M, N = 4096 * 1024, 480
    layouts = [
        (1, 256, 1),
        (2, 256, 1),
        (4, 256, 1),
        (8, 256, 1),
        (64, 256, 1),
        (256, 256, 1),
        (1, 512, 2),
    ]
    data = []
    for BLOCK_M, BLOCK_N, num_warps in layouts:
        t = run(M, N, BLOCK_M, BLOCK_N, num_warps)
        t_torch = run_torch(M, N)
        data.append(
            (M, N, BLOCK_M, BLOCK_N, num_warps, f"{t:.3f}ms", f"{t_torch:.3f}ms")
        )
    headers = ["M", "N", "BLOCK_M", "BLOCK_N", "num_warps", "time", "time_torch"]
    print(tabulate(data, headers=headers))


################################################################################
#
#   Benchmark the performance of different layout.


#       Group            M    N    BLOCK_M    BLOCK_N    num_warps  time      time_torch
#       ---------  -------  ---  ---------  ---------  -----------  --------  ------------
#       BLOCK_N    4194304  480          1         32            1  37.806ms  7.776ms
#       BLOCK_N    4194304  480          1         64            1  20.161ms  7.776ms
#       BLOCK_N    4194304  480          1        128            1  10.098ms  7.776ms
#       BLOCK_N    4194304  480          1        256            1  8.084ms   7.776ms
#       BLOCK_N    4194304  480          1        512            1  7.809ms   7.776ms
#       BLOCK_N    4194304  480          1       1024            1  7.792ms   7.776ms
#
#       BLOCK_M    4194304  480          1        512            1  7.811ms   7.776ms
#       BLOCK_M    4194304  480          2        512            1  7.853ms   7.777ms
#       BLOCK_M    4194304  480          4        512            1  7.816ms   7.775ms
#       BLOCK_M    4194304  480         32        512            1  25.503ms  7.776ms
#       BLOCK_M    4194304  480        128        512            1  27.666ms  7.776ms
#
#       num_warps  4194304  480          1        512            1  7.810ms   7.776ms
#       num_warps  4194304  480          1        512            2  7.815ms   7.777ms
#       num_warps  4194304  480          1        512            4  7.784ms   7.777ms
#       num_warps  4194304  480          1        512            8  7.739ms   7.775ms
#       num_warps  4194304  480          1        512           16  9.209ms   7.775ms
#
#   For a tall-thin-shape matrix,
#   1. Not much difference between 1D and 2D grids
#   2. BLOCK_N is the most important factor. Need to be big enough to
#       - hide the memory access latency,
#       - better coalescing memory access,
#       - and compute parallelism.
#   3. BLOCK_M is kind of irrelevant, as long as it's not too small.
#   4. Usually num_warps=1 gets the best performance.
#       - Large num_warps doesn't make it faster but usually slower.
#       - It might because of preventing more parallelism on M axis.
#
################################################################################
def benchmark_layout():
    """
    Test the performance of different layout.

    For a tall-thin-shape matrix,
    1. Not much difference between 1D and 2D grids
    2. BLOCK_N is the most important factor. Need to be big enough to
        - hide the memory access latency,
        - better coalescing memory access,
        - and compute parallelism.
    3. BLOCK_M is kind of irrelevant, as long as it's not too small.
    4. Usually num_warps=1 gets the best performance.
        - Large num_warps doesn't make it faster but usually slower.
        - It might because of preventing more parallelism on M axis.
    """
    M, N = 4096 * 1024, 480
    BLOCK_N_layouts = [
        (1, 32, 1),
        (1, 64, 1),
        (1, 128, 1),
        (1, 256, 1),
        (1, 512, 1),
        (1, 1024, 1),
    ]
    BLOCK_M_layouts = [
        (1, 512, 1),
        (2, 512, 1),
        (4, 512, 1),
        (32, 512, 1),
        (128, 512, 1),
    ]
    warps_layouts = [
        (1, 512, 1),
        (1, 512, 2),
        (1, 512, 4),
        (1, 512, 8),
        (1, 512, 16),
    ]
    layout_groups = {
        "BLOCK_N": BLOCK_N_layouts,
        "BLOCK_M": BLOCK_M_layouts,
        "num_warps": warps_layouts,
    }
    data = []
    for group, layouts in layout_groups.items():
        for BLOCK_M, BLOCK_N, num_warps in layouts:
            t = run(M, N, BLOCK_M, BLOCK_N, num_warps)
            t_torch = run_torch(M, N)
            data.append(
                (
                    group,
                    M,
                    N,
                    BLOCK_M,
                    BLOCK_N,
                    num_warps,
                    f"{t:.3f}ms",
                    f"{t_torch:.3f}ms",
                )
            )
        data.append([])
    headers = [
        "Group",
        "M",
        "N",
        "BLOCK_M",
        "BLOCK_N",
        "num_warps",
        "time",
        "time_torch",
    ]
    print(tabulate(data, headers=headers))


################################################################################
#
#   Auto-tuning
#
#   Test:
#       Auto-tuned: M: 4194304, N: 480, time: 7.786ms
#
#   Takeaway:
#   1. triton.Config has default values
#       - num_stages=3
#       - num_warps=4
#   2. grid = lambda META:
#       - META can access all the arguments of the kernel, but not values from triton.Config
#       - for example, if arguments don't include num_warps, it cannot be accessed
#   3. autotune seems fast and accurate. Comparable to hand-tuned layouts
#
################################################################################


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 128, "num_warps": 1}),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 256, "num_warps": 1}),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 512, "num_warps": 1}),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 1024, "num_warps": 1}),
    ],
    key=["N"],
)
@triton.jit
def add_kernel_auto_tuned(
    A,
    B,
    Out,
    M,
    N,
    stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # num_warps: NO need to define num_warps here. Likely defined inside triton.jit already.
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (off_m[:, None] < M) & (off_n[None, :] < N)
    off = off_m[:, None] * stride + off_n[None, :]
    a = tl.load(A + off, mask=mask, other=0)
    b = tl.load(B + off, mask=mask, other=0)
    c = a + b
    tl.store(Out + off, c, mask=mask)


class AddAutoTuned(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        assert a.is_contiguous()
        assert b.is_contiguous()
        out = torch.empty_like(a)
        M, N = a.shape
        # META is a dict that contains all the arguments passed to the kernel.
        grid = lambda META: (
            triton.cdiv(META["M"], META["BLOCK_M"]),
            triton.cdiv(META["N"], META["BLOCK_N"]),
        )
        add_kernel_auto_tuned[grid](a, b, out, M, N, a.stride(0))
        return out


def add_auto_tuned(a, b):
    return AddAutoTuned.apply(a, b)


def test_auto_tuned():
    M, N = 4096 * 1024, 480
    a = torch.randn(M, N, device="cuda")
    b = torch.randn(M, N, device="cuda")
    add_auto_tuned(a, b)

    event_start = torch.cuda.Event(enable_timing=True)
    event_end = torch.cuda.Event(enable_timing=True)
    event_start.record()
    N_ITER = 10
    for _ in range(N_ITER):
        add_auto_tuned(a, b)
    event_end.record()
    torch.cuda.synchronize()
    dur = event_start.elapsed_time(event_end) / N_ITER
    print(f"Auto-tuned: M: {M}, N: {N}, time: {dur:.3f}ms")


################################################################################
#
#   Heuristics
#       Heuristics: M: 4194304, N: 480, time: 7.788ms
#
#   Seems handy and sufficiently efficient if has a good heuristic.
#
################################################################################


@triton.heuristics(
    values={
        "BLOCK_M": lambda META: 1,
        "BLOCK_N": lambda META: triton.next_power_of_two(META["N"]),
        "num_warps": lambda META: 1,
    }
)
@triton.jit
def add_kernel_heuristics(
    A,
    B,
    Out,
    M,
    N,
    stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # num_warps: NO need to define num_warps here. Likely defined inside triton.jit already.
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (off_m[:, None] < M) & (off_n[None, :] < N)
    off = off_m[:, None] * stride + off_n[None, :]
    a = tl.load(A + off, mask=mask, other=0)
    b = tl.load(B + off, mask=mask, other=0)
    c = a + b
    tl.store(Out + off, c, mask=mask)


class AddHeuristics(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        assert a.is_contiguous()
        assert b.is_contiguous()
        out = torch.empty_like(a)
        M, N = a.shape
        # META is a dict that contains all the arguments passed to the kernel.
        grid = lambda META: (
            triton.cdiv(META["M"], META["BLOCK_M"]),
            triton.cdiv(META["N"], META["BLOCK_N"]),
        )
        add_kernel_auto_tuned[grid](a, b, out, M, N, a.stride(0))
        return out


def add_heuristics(a, b):
    return AddHeuristics.apply(a, b)


def test_heuristics():
    M, N = 4096 * 1024, 480
    a = torch.randn(M, N, device="cuda")
    b = torch.randn(M, N, device="cuda")
    add_heuristics(a, b)

    event_start = torch.cuda.Event(enable_timing=True)
    event_end = torch.cuda.Event(enable_timing=True)
    event_start.record()
    N_ITER = 10
    for _ in range(N_ITER):
        add_heuristics(a, b)
    event_end.record()
    torch.cuda.synchronize()
    dur = event_start.elapsed_time(event_end) / N_ITER
    print(f"Heuristics: M: {M}, N: {N}, time: {dur:.3f}ms")


def main():
    # test_correctness()
    # benchmark_1d_vs_2d_grid()
    # benchmark_layout()
    # test_auto_tuned()
    test_heuristics()


if __name__ == "__main__":
    main()
