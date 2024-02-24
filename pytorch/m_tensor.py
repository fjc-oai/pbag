import torch


def test_storage():
    t = torch.rand(4, 8)
    t2 = t[2:, :]
    assert t.data_ptr() != t2.data_ptr()
    assert t.storage().data_ptr() == t2.storage().data_ptr()


def test_seed():
    torch.manual_seed(7)
    tensors = [torch.rand(4, 8) for _ in range(10)]
    torch.manual_seed(7)
    tensors2 = [torch.rand(4, 8) for _ in range(10)]
    for t1, t2 in zip(tensors, tensors2):
        assert torch.equal(t1, t2)


def test_associative():
    types = [torch.float32, torch.float16]
    shape = (16, 16)
    for t in types:
        torch.manual_seed(7)
        a, b, c = [torch.rand(shape, dtype=t) for _ in range(3)]
        x = a + (b + c)
        y = (a + b) + c
        assert not torch.equal(x, y)
    print("Associative test passed!")
