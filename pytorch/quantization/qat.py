import torch


def bwd(swap: bool, swap_back: bool):
    torch.manual_seed(0)

    a = torch.randn(4, 4, requires_grad=True)
    b = torch.randn(4, 4, requires_grad=True)
    c = torch.randn(4, 4, requires_grad=True)

    if swap:
        a._swap_buff = a.data
        a.data = c.data

    loss = (a @ b).sum()

    if swap_back:
        a.data = a._swap_buff

    loss.backward()

    return loss, a, b, c


def test_qat():
    loss1, a1, b1, c1 = bwd(False, False)
    loss2, a2, b2, c2 = bwd(True, True)
    loss3, a3, b3, c3 = bwd(True, False)

    assert not torch.equal(loss1, loss2)
    assert torch.equal(a1, a2)
    assert torch.equal(b1, b2)
    assert torch.equal(a1.grad, a2.grad)
    assert torch.equal(b1.grad, b2.grad)

    assert not torch.equal(loss1, loss3)
    assert not torch.equal(a1, a3)
    assert torch.equal(b1, b3)
    assert torch.equal(a1.grad, a3.grad)
    assert not torch.equal(b1.grad, b3.grad)
    print("Test passed!")


if __name__ == "__main__":
    test_qat()
