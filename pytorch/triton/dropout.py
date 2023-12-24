# https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html#references
import tabulate
import torch

import triton
import triton.language as tl


@triton.jit
def _dropout(
    x_ptr,  # pointer to the input
    x_keep_ptr,  # pointer to a mask of 0s and 1s
    output_ptr,  # pointer to the output
    n_elements,  # number of elements in the `x` tensor
    p,  # probability that an element of `x` is changed to zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    # The line below is the crucial part, described in the paragraph above!
    output = tl.where(x_keep, x / (1 - p), 0.0)
    # Write-back output
    tl.store(output_ptr + offsets, output, mask=mask)


def dropout(x, x_keep, p):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output


def test_baseline():
    # Input tensor
    x = torch.randn(size=(10,)).cuda()
    # Dropout mask
    p = 0.5
    x_keep = (torch.rand(size=(10,)) > p).to(torch.int32).cuda()
    #
    output = dropout(x, x_keep=x_keep, p=p)
    print(
        tabulate.tabulate(
            [
                ["input"] + x.tolist(),
                ["keep mask"] + x_keep.tolist(),
                ["output"] + output.tolist(),
            ]
        )
    )


@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly prune it
    random = tl.rand(seed, offsets)
    tl.device_print("seeded", random)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output


def test_seeded_dropout():
    seed = 41
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = torch.randn(size=(10,)).cuda()
    # Compare this to the baseline - dropout mask is never instantiated!
    output = seeded_dropout(x, p=0.5, seed=123)
    output2 = seeded_dropout(x, p=0.5, seed=123)
    output3 = seeded_dropout(x, p=0.5, seed=512)

    print(
        tabulate.tabulate(
            [
                ["input"] + x.tolist(),
                ["output (seed = 123)"] + output.tolist(),
                ["output (seed = 123)"] + output2.tolist(),
                ["output (seed = 512)"] + output3.tolist(),
            ]
        )
    )


@triton.jit
def _maxtrix_dropout(
    x_ptr,
    output_ptr,
    stride,
    p,
    seeds,
    BLOCK_SIZE: tl.constexpr,
):
    pid1 = tl.program_id(0)
    pid2 = tl.program_id(1)
    seed = tl.load(seeds + pid1)
    print("seed", seed)
    start = pid1 * stride + pid2 * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < stride
    x = tl.load(x_ptr + start + offsets, mask=mask)
    random = tl.rand(seed, offsets)
    # Why this prints tons of lines, e.g. pid (0, 0, 0) idx ( 935) matrix: 0.987880
    # tl.device_print("matrix:", random)
    x_keep = random > p
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + start + offsets, output, mask=mask)


def matrix_dropout(x, p, seeds):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    assert x.dim() == 2
    grid = lambda meta: (x.size(0), triton.cdiv(x.size(1), meta["BLOCK_SIZE"]))
    _maxtrix_dropout[grid](x, output, x.stride(0), p, seeds, BLOCK_SIZE=1024)
    return output


def test_matrix_dropout():
    x = torch.randn(size=(10, 10)).cuda()
    seeds = torch.randint(100, (10,)).cuda()
    output = matrix_dropout(x, p=0.5, seeds=seeds)
    print(output)


def cmp():
    # THIS IS NOT WORKING....
    torch_seed = 41
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    x = torch.randn(4096, 8024).cuda()
    p = 0.5
    seeds = torch.randint(100, (x.size(0),)).cuda()

    matrix_output = matrix_dropout(x, p=p, seeds=seeds)
    seeded_outputs = []
    for i in range(x.size(0)):
        seeded_output = seeded_dropout(x[i], p=p, seed=seeds[i])
        seeded_outputs.append(seeded_output)
    seeded_outputs = torch.stack(seeded_outputs)
    print(f"Matrix dropout: {matrix_output}")
    print(f"Seeded dropout: {seeded_outputs}")
    assert torch.allclose(matrix_output, seeded_outputs)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["baseline", "seeded", "matrix", "cmp"])
    args = parser.parse_args()
    if args.mode == "baseline":
        test_baseline()
    elif args.mode == "seeded":
        test_seeded_dropout()
    elif args.mode == "matrix":
        test_matrix_dropout()
    elif args.mode == "cmp":
        cmp()
    else:
        raise ValueError(f"Unknown mode {args.mode}")


if __name__ == "__main__":
    main()
