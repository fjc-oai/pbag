import torch
import triton
import triton.language as tl


"""
Any better way to do this?????
"""
@triton.jit()
def slice_cat_kernel(
    X,
    INDEXES,
    OUT,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    off = tl.arange(0, BLOCK_SIZE)
    mask = off < N
    a = tl.load(X + off, mask=mask)
    b = tl.load(X + N + off, mask=mask)

    off = tl.arange(0, BLOCK_SIZE * 2)
    mask = off < N * 2
    indexes = tl.load(INDEXES + off, mask=mask)

    c = tl.interleave(a, b)
    c = tl.gather(c, indexes, 0)

    tl.store(OUT + off, c, mask=mask)


def test():
    x = torch.arange(480, dtype=torch.float32, device="cuda")
    out = torch.empty(480, dtype=torch.float32, device="cuda")
    idx = list(range(10))
    idx = [i * 2 for i in idx]
    idx2 = [i + 1 for i in idx]
    idx = idx + idx2
    indexes = torch.tensor(idx, dtype=torch.int32, device="cuda")
    N = x.numel() // 2
    BLOCK_SIZE = 256
    slice_cat_kernel[(1,)](x, indexes, out, N, BLOCK_SIZE)


test()
