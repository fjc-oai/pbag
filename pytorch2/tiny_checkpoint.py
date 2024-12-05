"""
This script demonstrates how to implement a toy version of PyTorch's activation
checkpointing.
 
- test_correctness() verifies that the toy implementation produces the same
  output and gradient as the no_op and torch implementations.

- test_peak_memory() compares the peak memory usage of the toy implementation,
  and confirms it indeeds use less memory and equals to torch's implementation
  when use_reentrant=True.

It also includes options to visualize the computation graph, either by printing
through dot.render() or by printing through print_graph().

"""
from collections import defaultdict

import torch
import torch.utils.checkpoint as torch_checkpoint
from torchviz import make_dot

ENABLE_COMPUTATION_GRAPH_VIZ_DOT = False
ENABLE_COMPUTATION_GRAPH_VIZ_PRINT = False

class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, *args):
        ctx.run_function = run_function
        ctx.save_for_backward(*args)
        with torch.no_grad():
            outputs = run_function(*args)
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs = ctx.saved_tensors
        run_function = ctx.run_function
        with torch.enable_grad():
            ####################################################################
            # 
            # IMPORTANT: Detach the input. So to ensure this backward() function
            # only computes the gradient till to the input of this method, but
            # not further along the chain.
            # 
            ####################################################################
            inputs = tuple(input_.detach().requires_grad_(True) for input_ in inputs)
            outputs = run_function(*inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        torch.autograd.backward(outputs, grad_outputs)
        grad_inputs = tuple(input_.grad if input_.requires_grad else None for input_ in inputs)
        return (None,) + grad_inputs  # None for run_function


def get_checkpoint_fn(mode):
    def checkpoint_no_op(run_function, *args):
        return run_function(*args)

    def checkpoint_toy(run_function, *args):
        return CheckpointFunction.apply(run_function, *args)

    if mode == "no_op":
        return checkpoint_no_op
    elif mode == "toy":
        return checkpoint_toy
    elif mode == "torch":
        from functools import partial

        # Setting use_reentrant=False provides better memory usage, whereas
        # setting use_reentrant=True gives identical memory usage as the toy
        # implementation
        return partial(torch_checkpoint.checkpoint, use_reentrant=True)

def print_graph(fn, indent=0):
    if fn is None:
        return
    print(' ' * indent + f'-> {type(fn).__name__}')
    for next_fn, _ in fn.next_functions:
        print_graph(next_fn, indent + 4)

def run_fwd_bwd(mode):
    class Block(torch.nn.Module):
        def __init__(self, d_model):
            super(Block, self).__init__()
            self.layer1 = torch.nn.Linear(d_model, d_model)
            self.layer2 = torch.nn.Linear(d_model, d_model)
            self.layer3 = torch.nn.Linear(d_model, d_model)
            self.layer4 = torch.nn.Linear(d_model, d_model)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = torch.relu(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = torch.relu(x)
            return x

    class Model(torch.nn.Module):
        def __init__(self, d_model):
            super(Model, self).__init__()
            # PyTorch’s checkpoint function requires that at least one of the
            # inputs to the checkpointed function has requires_grad=True! !!
            self.proj_in = torch.nn.Linear(d_model, d_model)
            self.blocks = torch.nn.ModuleList([Block(d_model) for _ in range(4)])
            self.proj_out = torch.nn.Linear(d_model, d_model)
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)

        def forward(self, x):
            x = self.proj_in(x)

            checkpoint_fn = get_checkpoint_fn(mode)
            for block in self.blocks:
                x = checkpoint_fn(block, x)
            x = self.proj_out(x)
            return x

    torch.manual_seed(7)

    d_model = 1024
    bs = 4096 * 2

    model = Model(d_model).cuda()
    x = torch.randn(bs, d_model).cuda().requires_grad_(True)
    target = torch.randn(bs, d_model).cuda()
    y = model(x)
    loss = torch.nn.MSELoss()(y, target)

    if ENABLE_COMPUTATION_GRAPH_VIZ_DOT:
        dot = make_dot(loss, params={"x": x, "y": y, "loss": loss})
        dot.format = "pdf"
        dot.render(f"{mode}_checkpoint")
    if ENABLE_COMPUTATION_GRAPH_VIZ_PRINT:
        print_graph(loss.grad_fn)
    loss.backward()
    return x, y, loss, model


def cmp(lhs, rhs):
    for k in ["x", "y", "loss"]:
        assert torch.equal(lhs[k], rhs[k]), f"lhs[{k}] != rhs[{k}]"
    lhs_model = lhs["model"]
    rhs_model = rhs["model"]
    for lhs_param, rhs_param in zip(lhs_model.parameters(), rhs_model.parameters()):
        assert torch.equal(lhs_param, rhs_param), f"lhs_param != rhs_param"
        assert torch.equal(lhs_param.grad, rhs_param.grad), f"lhs_param.grad != rhs_param.grad"


def test_correctness():
    modes = ["no_op", "toy", "torch"]
    res = defaultdict(dict)

    for mode in modes:
        x, y, loss, model = run_fwd_bwd(mode)
        res[mode]["x"] = x
        res[mode]["y"] = y
        res[mode]["loss"] = loss
        res[mode]["model"] = model

    for mode in modes[1:]:
        cmp(res[modes[0]], res[mode])
        print(f"{mode} passed")


def test_peak_memory():
    modes = ["no_op", "toy", "torch"]
    for mode in modes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x, y, loss, model = run_fwd_bwd(mode)
        del x, y, loss, model
        print(f"{mode} peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")


if __name__ == "__main__":
    test_correctness()
    test_peak_memory()
