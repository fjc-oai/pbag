"""
- capture(): use torch.cuda.graph() to capture a graph of a function.
    - It seems to be a bug that graph capture requieres to manually run matmul() before the capturing.

- make_callable(): use torch.cuda.make_graphed_callables() to make a callable from a function.
    - It seems to require all the inputs to require_grad=True.

- if_else_capture(): graph capture does not support if-else.

- logging_capture(): all the print statements are emmitted during graph capture and replay.

- assertion_capture(): use torch.inverse() to simulate if-else to trigger assertion. Turns out not supported by graph capture.
"""

import time

import torch


def compute(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    x = torch.matmul(x, y)
    y = torch.matmul(y, x)
    t = x + y
    out.copy_(t)


def capture():
    D = 512
    s_x = torch.randn(D, D, device="cuda")
    s_y = torch.randn(D, D, device="cuda")
    s_out = torch.randn(D, D, device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        compute(s_x, s_y, s_out)

    x = torch.rand_like(s_x)
    y = torch.rand_like(s_y)
    expected = torch.rand_like(s_out)
    compute(x, y, expected)

    s_x.copy_(x)
    s_y.copy_(y)
    g.replay()
    out = s_out.clone()
    assert torch.allclose(out, expected)


def compute2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = torch.matmul(x, y)
    y = torch.matmul(y, x)
    return x + y


def make_callable():
    D = 512
    s_x = torch.randn(D, D, device="cuda", requires_grad=True)
    s_y = torch.randn(D, D, device="cuda", requires_grad=True)
    g = torch.cuda.make_graphed_callables(compute2, (s_x, s_y))

    x = torch.rand_like(s_x)
    y = torch.rand_like(s_y)
    expected = compute2(x, y)

    out = g(x, y)
    assert torch.equal(out, expected)


def if_else_compute(x: torch.Tensor, y: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
    """
      File "/root/code/pbag/pytorch/m_cuda_graph.py", line 95, in if_else_compute
        if torch.all(x > 0):
    RuntimeError: CUDA error: operation not permitted when stream is capturing
    """

    if torch.all(x > 0):
        res.copy_(x)
    else:
        res.copy_(y)


def if_else_capture():
    D = 4
    s_x = torch.randn(D, D, device="cuda")
    s_y = torch.randn(D, D, device="cuda")
    s_out = torch.randn(D, D, device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        if_else_compute(s_x, s_y, s_out)


def logging_compute(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    print(f"running logging_compute")
    out = x + y


def logging_capture():
    D = 4
    s_x = torch.randn(D, D, device="cuda")
    s_y = torch.randn(D, D, device="cuda")
    s_out = torch.randn(D, D, device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        logging_compute(s_x, s_y, s_out)
    torch.cuda.synchronize()
    print("Now replaying the graph...")
    g.replay()
    print("Replay done")


# RuntimeError: CUDA error: operation not permitted when stream is capturing
def _check_eq(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    diff = lhs - rhs
    t = diff.abs().max()
    t = torch.where(t == 0, 1, 0)
    m = torch.eye(2, device="cuda")
    m[0, 0] = t
    torch.inverse(m)


def assertion_in_compute(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor, lhs: torch.Tensor, rhs: torch.Tensor
) -> torch.Tensor:
    x = torch.matmul(x, y)
    y = torch.matmul(y, x)
    _check_eq(lhs, rhs)
    t = x + y
    out.copy_(t)


def assertion_capture():
    D = 4
    s_x = torch.randn(D, D, device="cuda")
    s_y = torch.randn(D, D, device="cuda")
    s_out = torch.randn(D, D, device="cuda")
    s_lhs = torch.randn(D, D, device="cuda")
    s_rhs = s_lhs.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        assertion_in_compute(s_x, s_y, s_out, s_lhs, s_rhs)

    try:
        lhs = torch.rand_like(s_lhs)
        rhs = torch.rand_like(s_rhs)
        s_lhs.copy_(lhs)
        s_rhs.copy_(rhs)
        g.replay()
    except RuntimeError as e:
        print(e)
        print("Replay failed")
    print("assertion_capture done")


def main():
    # https://github.com/pytorch/pytorch/issues/99397
    x = torch.randn(512, 512, device="cuda")
    y = torch.randn(512, 512, device="cuda")
    torch.matmul(x, y)

    m = torch.eye(2, device="cuda")
    torch.inverse(m)

    capture()
    make_callable()
    # if_else_capture()
    logging_capture()
    # assertion_capture()


if __name__ == "__main__":
    main()
