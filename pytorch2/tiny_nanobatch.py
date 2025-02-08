"""
This script demonstrates how to implement a toy version of nano-batching to
reduce memory usage.

## Memory analysis - corse grained ## 
Params:
    - proj_in: 4*(1024**2) = 4MB
    - blocks: 4*(4*1024**2)*16 = 256MB
    - proj_out: 4*(1024**2) = 4MB
    - total: 264MB

Grads:
    - total: 264MB

Acts common:
    - proj_in and proj_out: 4*(8192*1024)*2 = 64MB

** Vanilla mode **
    - Acts fwd: 4*(8192*1024)*4*8 = 1024MB
    - Total: 264*2 + 64 + 1024 = 1616MB

** Checkpoint mode **
    - Acts fwd: 4*(8192*1024)*8 = 256MB
    - Acts bwd: 4*(8192*1024)*4 = 128MB
    - Total: 264*2 + 64 + 256 + 128 = 712MB

** Nano mode **
    - Acts fwd: 4*(8192*1024)*8 = 256MB
    - Acts bwd: 4*(512*1024)*4 = 8MB    
    - Total: 264*2 + 64 + 256 + 8 = 592MB

## Execution Results ##

$ py tiny_nanobatch.py
################################################################################
Testing correctness
vanilla vs ckpt: match
vanilla vs nano: match
vanilla vs nano2: match
################################################################################
Testing memory usage
Peak memory usage [vanilla]: 1516.14 MB
Memory profile dumped to /tmp/mem_profile_vanilla.pickle
Peak memory usage [ckpt]: 880.14 MB
Memory profile dumped to /tmp/mem_profile_ckpt.pickle
Peak memory usage [nano]: 748.14 MB
Memory profile dumped to /tmp/mem_profile_nano.pickle
Peak memory usage [nano2]: 748.14 MB
Memory profile dumped to /tmp/mem_profile_nano2.pickle

"""

from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.checkpoint import checkpoint


@dataclass
class Config:
    mode: Literal["vanilla", "ckpt", "nano", "nano2"] = "vanilla"
    bs: int = 4096 * 2
    d_model: int = 1024
    n_layers: int = 8


CHUNK_SIZE = 512


@dataclass
class TestConfig:
    cmp_results: bool = True
    profile_memory: bool = False

    def __post_init__(self):
        assert not (self.cmp_results and self.profile_memory)


class Chunked(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fn):
        ctx.fn = fn
        ctx.save_for_backward(x)

        results = []
        n_chunks = x.size(0) // CHUNK_SIZE
        for chunked_x in x.chunk(n_chunks, dim=0):
            results.append(fn(chunked_x))
        return torch.cat(results, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        fn = ctx.fn
        d_x = []
        with torch.enable_grad():
            assert x.shape[0] == grad_output.shape[0]
            for i in range(0, x.size(0), CHUNK_SIZE):
                chunked_x = x[i : i + CHUNK_SIZE].detach().requires_grad_(True)
                act = fn(chunked_x)
                act.backward(grad_output[i : i + CHUNK_SIZE].detach())
                d_x.append(chunked_x.grad)
        d_x = torch.cat(d_x, dim=0)
        return d_x, None


class Block(torch.nn.Module):
    def __init__(self, config: Config):
        super(Block, self).__init__()
        self.config = config
        self.layer1 = torch.nn.Linear(config.d_model, config.d_model)
        self.layer2 = torch.nn.Linear(config.d_model, config.d_model)
        self.layer3 = torch.nn.Linear(config.d_model, config.d_model)
        self.layer4 = torch.nn.Linear(config.d_model, config.d_model)

    def forward(self, x):
        if self.config.mode == "vanilla":
            return self._vanilla_fwd(x)
        elif self.config.mode == "ckpt":
            return self._ckpt_fwd(x)
        elif self.config.mode == "nano":
            return self._nano_fwd(x)
        elif self.config.mode == "nano2":
            return self._nano2_fwd(x)
        else:
            raise ValueError(f"Invalid mode: {self.config.mode}")

    def _vanilla_fwd(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.relu(x)
        return x

    def _ckpt_fwd(self, x):
        x = checkpoint(self._vanilla_fwd, x)
        return x

    def _nano_fwd(self, x):
        n_chunks = x.size(0) // CHUNK_SIZE
        res = []
        for chunked_x in x.chunk(n_chunks, dim=0):
            chunked_x = checkpoint(self._vanilla_fwd, chunked_x)
            res.append(chunked_x)
        x = torch.cat(res, dim=0)
        return x

    def _nano2_fwd(self, x):
        return Chunked.apply(x, self._vanilla_fwd)


class Model(torch.nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        self.config = config
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
        for block in self.blocks:
            x = block(x)
        x = self.proj_out(x)
        return x


def run_fwd_bwd(config: Config, test_config: TestConfig):
    if test_config.profile_memory:
        torch.cuda.memory._record_memory_history(enabled=None)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.memory._record_memory_history()
    torch.manual_seed(7)
    model = Model(config).cuda()
    x = torch.randn(config.bs, config.d_model).cuda()
    target = torch.randn(config.bs, config.d_model).cuda()
    y = model(x)
    loss = torch.nn.MSELoss()(y, target)
    loss.backward()
    if test_config.profile_memory:
        print(
            f"Peak memory usage [{config.mode}]: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB"
        )
        torch.cuda.memory._dump_snapshot(f"/tmp/mem_profile_{config.mode}.pickle")
        print(f"Memory profile dumped to /tmp/mem_profile_{config.mode}.pickle")
    return {"x": x, "y": y, "loss": loss, "model": model}

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
    vanilla_config = Config(mode="vanilla")
    ckpt_config = Config(mode="ckpt")
    nano_config = Config(mode="nano")
    nano2_config = Config(mode="nano2")

    if test_config.cmp_results:
        ckpt_res = run_fwd_bwd(ckpt_config, test_config)
        vanilla_res = run_fwd_bwd(vanilla_config, test_config)
        nano_res = run_fwd_bwd(nano_config, test_config)
        nano2_res = run_fwd_bwd(nano2_config, test_config)
        print(f"vanilla vs ckpt: {colorize_result(cmp_results(vanilla_res, ckpt_res))}")
        print(f"vanilla vs nano: {colorize_result(cmp_results(vanilla_res, nano_res))}")
        print(f"vanilla vs nano2: {colorize_result(cmp_results(vanilla_res, nano2_res))}")

    elif test_config.profile_memory:
        run_fwd_bwd(vanilla_config, test_config)
        run_fwd_bwd(ckpt_config, test_config)
        run_fwd_bwd(nano_config, test_config)
        run_fwd_bwd(nano2_config, test_config)


if __name__ == "__main__":
    print("#" * 80)
    print("Testing correctness")
    test_config = TestConfig(cmp_results=True, profile_memory=False)
    test(test_config)

    print("#" * 80)
    print("Testing memory usage")
    test_config = TestConfig(cmp_results=False, profile_memory=True)
    test(test_config)
