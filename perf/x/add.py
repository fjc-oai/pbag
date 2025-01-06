import torch
import triton
import triton.language as tl


@triton.jit
def add_fwd(
    X,
    Y,
    Out,
    N,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    start_pos = pid * BLOCK_SIZE_N
    off = tl.arange(0, BLOCK_SIZE_N)
    pos = start_pos + off
    mask = pos < N
    x = tl.load(X + pos, mask=mask, other=0)
    y = tl.load(Y + pos, mask=mask, other=0)
    out = x + y
    tl.store(Out + pos, out, mask=mask)


@triton.jit
def add_bwd(
    D_Out,
    D_X,
    D_Y,
    N,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    start_pos = pid * BLOCK_SIZE_N
    off = tl.arange(0, BLOCK_SIZE_N)
    pos = start_pos + off
    mask = pos < N
    out = tl.load(D_Out + pos, mask=mask, other=0)
    tl.store(D_X + pos, out, mask=mask)
    tl.store(D_Y + pos, out, mask=mask)


class Add(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        assert x.shape == y.shape
        assert x.is_contiguous() and y.is_contiguous()
        out = torch.empty_like(x)
        N = x.numel()
        BLOCK_SIZE_N = 128
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_N"]),)
        add_fwd[grid](x, y, out, N, BLOCK_SIZE_N)
        ctx.save_for_backward(x, y)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = (
            grad_out.contiguous()
        )  # !!!!! This could be non-contiguous for some reason. Debugged for an hour. !!!!!
        x, y = ctx.saved_tensors
        d_x = torch.empty_like(x)
        d_y = torch.empty_like(y)
        assert grad_out.is_contiguous()
        assert d_x.is_contiguous() and d_y.is_contiguous()
        N = x.numel()
        BLOCK_SIZE_N = 128
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_N"]),)
        add_bwd[grid](grad_out, d_x, d_y, N, BLOCK_SIZE_N)
        return (d_x, d_y)


def validate():
    torch.manual_seed(7)
    size = 8
    x = torch.rand(size, device="cuda").requires_grad_()
    y = torch.rand(size, device="cuda").requires_grad_()

    x2 = x.clone().detach().requires_grad_()
    y2 = y.clone().detach().requires_grad_()

    out = Add.apply(x, y)
    out.sum().backward()

    out2 = x2 + y2
    out2.sum().backward()

    assert torch.equal(out, out2)
    assert torch.equal(x.grad, x2.grad)
    assert torch.equal(y.grad, y2.grad)
    print("Validation successful!")


def profile():
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        # with_stack=True, # Otherwise very easy to cause invalid char in trace.json
    ) as prof:
        torch.manual_seed(7)
        size = 8
        x = torch.rand(size, device="cuda").requires_grad_()
        y = torch.rand(size, device="cuda").requires_grad_()
        out = Add.apply(x, y)
        out.sum().backward()
    prof.export_chrome_trace("trace.json")


if __name__ == "__main__":
    # validate()
    profile()
