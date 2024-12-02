import torch


def add(name: str, *tensor: torch.Tensor):
    n = len(tensor)
    res = torch.add(*tensor)
    return n, res


def test_varargs():
    n, res = add("test", torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]))
    assert n == 2
    assert torch.equal(res, torch.tensor([5, 7, 9]))
