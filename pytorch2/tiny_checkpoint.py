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
from dataclasses import dataclass
from typing import Literal

import torch
import torch.utils.checkpoint as torch_checkpoint
from torchviz import make_dot


@dataclass
class ModelConfig:
    d_model: int = 1024
    bs: int = 4096 * 2
    n_layers: int = 4
    mode: Literal["no_op", "toy", "torch"] = "no_op"


@dataclass
class TestConfig:
    cmp_results: bool = True
    profile_memory: bool = False

    enable_computation_graph_viz_dot: bool = False
    enable_computation_graph_viz_print: bool = False

    def __post_init__(self):
        assert not (self.cmp_results and self.profile_memory)


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
    print(" " * indent + f"-> {type(fn).__name__}")
    for next_fn, _ in fn.next_functions:
        print_graph(next_fn, indent + 4)


class Block(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super(Block, self).__init__()
        self.config = config
        self.layer1 = torch.nn.Linear(config.d_model, config.d_model)
        self.layer2 = torch.nn.Linear(config.d_model, config.d_model)
        self.layer3 = torch.nn.Linear(config.d_model, config.d_model)
        self.layer4 = torch.nn.Linear(config.d_model, config.d_model)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.relu(x)
        return x


class Model(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super(Model, self).__init__()
        self.config = config
        # PyTorch’s checkpoint function requires that at least one of the
        # inputs to the checkpointed function has requires_grad=True! !!
        self.proj_in = torch.nn.Linear(config.d_model, config.d_model)
        self.blocks = torch.nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.proj_out = torch.nn.Linear(config.d_model, config.d_model)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.proj_in(x)

        checkpoint_fn = get_checkpoint_fn(self.config.mode)
        for block in self.blocks:
            x = checkpoint_fn(block, x)
        x = self.proj_out(x)
        return x


def run_fwd_bwd(model_config: ModelConfig, test_config: TestConfig):
    if test_config.profile_memory:
        torch.cuda.memory._record_memory_history(enabled=None)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.memory._record_memory_history()
    torch.manual_seed(7)
    model = Model(model_config).cuda()
    x = torch.randn(model_config.bs, model_config.d_model).cuda().requires_grad_(True)
    target = torch.randn(model_config.bs, model_config.d_model).cuda()
    y = model(x)
    loss = torch.nn.MSELoss()(y, target)

    if test_config.enable_computation_graph_viz_dot:
        dot = make_dot(loss, params={"x": x, "y": y, "loss": loss})
        dot.format = "pdf"
        dot.render(f"{model_config.mode}_checkpoint")
    if test_config.enable_computation_graph_viz_print:
        print_graph(loss.grad_fn)
    loss.backward()

    if test_config.profile_memory:
        print(
            f"Peak memory usage: {model_config.mode}: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB"
        )
        torch.cuda.memory._dump_snapshot(f"/tmp/mem_profile_{model_config.mode}.pickle")
        print(f"Memory profile dumped to /tmp/mem_profile_{model_config.mode}.pickle")
    return {
        "x": x,
        "y": y,
        "loss": loss,
        "model": model,
    }


def cmp_tensors(lhs, rhs) -> bool:
    diff = (lhs - rhs).abs()
    rel_diff = diff / (lhs.abs() + rhs.abs() + 1e-6)
    return f"max diff: {diff.max()}, max rel diff: {rel_diff.max()}"


def cmp_results(lhs, rhs) -> bool:
    for k in ("x", "y", "loss"):
        if not torch.allclose(lhs[k], rhs[k], rtol=0.05, atol=1e-6):
            print(f"{k} mismatch: {cmp_tensors(lhs[k], rhs[k])}")
            return False
    lhs_model = lhs["model"]
    rhs_model = rhs["model"]
    for lhs_param, rhs_param in zip(lhs_model.parameters(), rhs_model.parameters()):
        if not torch.equal(lhs_param, rhs_param):
            print("param mismatch")
            return False
        if not torch.allclose(lhs_param.grad, rhs_param.grad, rtol=0.05, atol=1e-6):
            print(f"grad mismatch: {cmp_tensors(lhs_param.grad, rhs_param.grad)}")
            return False
    return True


def colorize(text, color):
    return f"\033[{color}m{text}\033[0m"


def colorize_result(match: bool) -> str:
    if match:
        return colorize("match", "32")
    else:
        return colorize("mismatch", "31")


def test(test_config: TestConfig):
    vanilla_config = ModelConfig(mode="no_op")
    toy_config = ModelConfig(mode="toy")
    torch_config = ModelConfig(mode="torch")

    if test_config.cmp_results:
        vanilla_res = run_fwd_bwd(vanilla_config, test_config)
        toy_res = run_fwd_bwd(toy_config, test_config)
        torch_res = run_fwd_bwd(torch_config, test_config)
        print(f"vanilla vs toy: {colorize_result(cmp_results(vanilla_res, toy_res))}")
        print(f"vanilla vs torch: {colorize_result(cmp_results(vanilla_res, torch_res))}")
    elif test_config.profile_memory:
        run_fwd_bwd(vanilla_config, test_config)
        run_fwd_bwd(toy_config, test_config)
        run_fwd_bwd(torch_config, test_config)


if __name__ == "__main__":
    print("#" * 80)
    print("Testing correctness")
    test_config = TestConfig(cmp_results=True, profile_memory=False)
    test(test_config)

    print("#" * 80)
    print("Testing memory usage")
    test_config = TestConfig(cmp_results=False, profile_memory=True)
    test(test_config)
