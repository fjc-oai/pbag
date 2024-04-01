"""
Toy implementation of PyTorch DistributedDataParallel (DPP) https://arxiv.org/abs/2006.15704

1. PyTorch launch multiple processes
2. Nccl collective communication group, and participate in collective communication
3. PyTorch DDP example
4. Toy DPP implementation
  - constructor: broadcast the model, register hooks
  - forward: forward pass
  - backward: triggers the hook and update grads (Note that, PyTorch DPP uses all_reduce(SUM)/world_size to average grads!!!)
5. Steps
  1. all_reduce grads in each param (naive implementation)
  2. verify correctness (compare with PyTorch DDP in cmp mode)
  3. bucketized all_reduce grads
  4. round-robin group
  5. use separate stream
"""

import argparse
import os
from contextlib import contextmanager
from typing import Literal

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

USE_GPU = torch.cuda.is_available()
PG_BACKEND: Literal["nccl", "gloo"] = "nccl" if USE_GPU else "gloo"
N_STEPS = 10
LR = 1.0


def rank_print(msg):
    print(f"Rank{dist.get_rank()}: {msg}")


def rank0_print(msg):
    if dist.get_rank() == 0:
        print(msg)


def launch(fn, world_size):
    processes = []
    mp.set_start_method(
        "spawn"
    )  # Otherise some unexpected crash happens. TODO: investigate the difference
    for rank in range(world_size):
        p = mp.Process(target=fn, args=(world_size, rank))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def _pg_setup(world_size, rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(PG_BACKEND, rank=rank, world_size=world_size)


def _pg_cleanup():
    dist.destroy_process_group()


@contextmanager
def use_pg(world_size, rank):
    _pg_setup(world_size=world_size, rank=rank)
    yield
    _pg_cleanup()


def pg_demo(world_size, rank):
    rank_print(f"Hello from rank {rank}/{world_size}")
    with use_pg(world_size=world_size, rank=rank):
        tensor = torch.tensor([rank])
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        rank_print(f"Rank {rank} has data {tensor}")


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(10, 5)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x

    def to_str(self, label):
        import tabulate

        table = []
        for name, param in self.named_parameters():
            table.append([name, torch.norm(param).item()])
        msg = "*" * 20 + f"{label} at rank{dist.get_rank()}" + "*" * 20
        msg += tabulate.tabulate(table, headers=["Param", "Norm"])
        msg = "\n" + msg + "\n"
        return msg


def pytorch_dpp_example(world_size, rank):
    with use_pg(world_size=world_size, rank=rank):
        torch.manual_seed(rank)
        model = ToyModel()
        dpp_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank] if USE_GPU else None
        )
        rank0_print(model.to_str("Before Training"))
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(dpp_model.parameters(), lr=LR)
        for step in range(N_STEPS):
            optimizer.zero_grad()
            input = torch.randn(20, 10)
            output = dpp_model(input)
            labels = torch.randn(20, 5)
            loss = loss_fn(output, labels)
            rank_print(f"Rank {rank} step {step} has loss {loss}")
            loss.backward()
            optimizer.step()
        rank0_print(model.to_str("After Training"))


class _Bucket:
    def __init__(self, param_names):
        self._param_names = param_names
        self._grad_ready_params = {}
        self._buffer = None
        self._handler = None

    def grad_ready(self, name, param):
        self._grad_ready_params[name] = param
        if len(self._grad_ready_params) == len(self._param_names):
            self._reduce_grads()
            return True
        return False

    def _reduce_grads(self):
        grads = [param.grad for param in self._grad_ready_params.values()]
        self._buffer = self._pack_to_buffer(grads)
        self._handler = dist.all_reduce(self._buffer, op=dist.ReduceOp.SUM, async_op=True)

    def wait(self):
        self._handler.wait()
        grads = [param.grad for param in self._grad_ready_params.values()]
        self._unpack_from_buffer(self._buffer, grads)
        for grad in grads:
            grad /= dist.get_world_size()
        self._grad_ready_params.clear()
        self._handler = None

    @staticmethod
    def _pack_to_buffer(tensors):
        total_size = sum(t.numel() for t in tensors)
        buffer = torch.zeros(total_size, dtype=tensors[0].dtype, device=tensors[0].device)
        offset = 0
        for t in tensors:
            numel = t.numel()
            buffer[offset : offset + numel].copy_(t.view(-1))
            offset += numel
        return buffer

    @staticmethod
    def _unpack_from_buffer(buffer, tensors):
        offset = 0
        for t in tensors:
            numel = t.numel()
            t.copy_(buffer[offset : offset + numel].view_as(t))
            offset += numel
        return tensors


class _Reducer:
    def __init__(self, module, bucket_size=2):
        self._module = module
        self._bucket_size = bucket_size
        self._param_to_bucket = {}
        self._buckets = []
        self._n_buckets = 0
        self._n_ready_buckets = 0
        self._init_buckets()

    def _init_buckets(self):
        for name, param in self._module.named_parameters():
            if param.requires_grad:
                if (
                    len(self._buckets) == 0
                    or len(self._buckets[-1]._param_names) == self._bucket_size
                ):
                    self._buckets.append(_Bucket([]))
                self._buckets[-1]._param_names.append(name)
                self._param_to_bucket[name] = self._buckets[-1]
        self._n_buckets = len(self._buckets)

    def grad_ready(self, name, param):
        bucket = self._param_to_bucket[name]
        ready = bucket.grad_ready(name, param)
        self._n_ready_buckets += ready
        if self._n_ready_buckets == self._n_buckets:
            for bucket in self._buckets:
                bucket.wait()
            self._n_ready_buckets = 0


class _NaiveReducer:
    def __init__(self, module):
        self._module = module
        self._n_params = len(list(module.parameters()))
        self._grad_ready_params = {}

    def grad_ready(self, name, param):
        self._grad_ready_params[name] = param
        if len(self._grad_ready_params) == self._n_params:
            handlers = []
            for name, param in self._module.named_parameters():
                grad = param.grad
                handler = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)
                grad /= dist.get_world_size()
                handlers.append(handler)
            for handler in handlers:
                handler.wait()
            self._grad_ready_params.clear()


class ToyDPP(torch.nn.Module):
    def __init__(self, module, bucket_size=2):
        super().__init__()
        self._module = module
        if bucket_size == None:
            self._reducer = _NaiveReducer(module)
        else:
            self._reducer = _Reducer(module, bucket_size)
        self._broadcast_module(module)
        for name, param in self._module.named_parameters():
            param.register_post_accumulate_grad_hook(self._autograd_hook(name))

    def forward(self, *inputs, **kwargs):
        return self._module(*inputs, **kwargs)

    def _broadcast_module(self, module):
        for p in module.parameters():
            dist.broadcast(p, 0)

    def _autograd_hook(self, name):
        def hook(param):
            self._reducer.grad_ready(name, param)

        return hook


def toy_dpp_example(world_size, rank):
    with use_pg(world_size=world_size, rank=rank):
        torch.manual_seed(rank)
        model = ToyModel()
        dpp_model = ToyDPP(model)
        rank_print(model.to_str("Before Training"))
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(dpp_model.parameters(), lr=LR)
        for step in range(N_STEPS):
            optimizer.zero_grad()
            input = torch.randn(20, 10)
            output = dpp_model(input)
            labels = torch.randn(20, 5)
            loss = loss_fn(output, labels)
            rank_print(f"Rank {rank} step {step} has loss {loss}")
            loss.backward()
            optimizer.step()
        rank_print(model.to_str("After Training"))


def cmp_examples(world_size, rank):
    if rank == 0:
        print(">>>> Now running PyTorch DDP Example")
    pytorch_dpp_example(world_size, rank)
    if rank == 0:
        print(">>>> Now running Toy DPP Example")
    toy_dpp_example(world_size, rank)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument(
        "--mode", type=str, default="dpp", choices=["pg_demo", "pytorch_dpp", "toy_dpp", "cmp"]
    )
    args = parser.parse_args()

    mode_map = {
        "pg_demo": pg_demo,
        "pytorch_dpp": pytorch_dpp_example,
        "toy_dpp": toy_dpp_example,
        "cmp": cmp_examples,
    }
    fn = mode_map[args.mode]
    launch(fn, args.world_size)


if __name__ == "__main__":
    main()
