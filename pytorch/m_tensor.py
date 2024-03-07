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


def test_backward():
    def get_tensors():
        torch.manual_seed(7)
        w = torch.rand(4, 8, requires_grad=True)
        x = torch.rand(8, 4, requires_grad=True)
        tgt = torch.rand(4, 4, requires_grad=True)
        return w, x, tgt

    w, x, tgt = get_tensors()
    y = w @ x
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(y, tgt)
    loss.backward()

    w_, x_, tgt_ = get_tensors()
    y_ = w_ @ x_
    loss_ = loss_fn(y_, tgt_)
    loss_.data = loss_.data * 2
    loss_.backward()
    assert torch.equal(loss * 2, loss_)
    for t1, t2 in zip([w, x, tgt], [w_, x_, tgt_]):
        assert torch.equal(t1, t2)
        assert torch.equal(t1.grad, t2.grad)
    print("Backward test passed!")
