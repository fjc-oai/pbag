import torch


# this is a sca
class ScalarMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, multiplier):
        ctx.multiplier = multiplier
        ctx.t = t
        ctx.save_for_backward(t)
        return t * multiplier
    
    @staticmethod
    def backward(ctx, grad_output):
        assert ctx.t is ctx.saved_tensors[0]
        return grad_output * ctx.multiplier, None
    
def test_cutomize_autograd_function():
    with torch.autograd.set_multithreading_enabled(False):
        t = torch.tensor([3, 5], requires_grad=True, dtype=torch.float32)
        r = t * 2
        r.backward(torch.tensor([1, 1], dtype=torch.float32))
        print(f"{t=}, {t.grad=}")
        
        t_ = t.clone().detach().requires_grad_(True)
        r_ = ScalarMul.apply(t_, 2.0)
        r_.backward(torch.tensor([1, 1], dtype=torch.float32))
        print(f"{t_=}, {t_.grad=}")
        assert torch.equal(t, t_)
        assert torch.equal(t.grad, t_.grad)
        print("test pass")
