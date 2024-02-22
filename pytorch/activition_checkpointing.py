import torch


def build_tensors():
    torch.manual_seed(0)
    x = torch.rand(2, 2)
    w_s = [torch.rand(2, 2, requires_grad=True) for _ in range(6)]
    return x, w_s


def foward():
    x, w_s = build_tensors()
    y_s = [x]
    for w in w_s:
        y = w @ y_s[-1]
        y_s.append(y)
    loss = torch.sum(y_s[-1])
    y_s[4].retain_grad()
    loss.backward()
    return x, w_s, y_s[2].clone().detach(), y_s[4].grad.clone().detach()


def recompute(y_2_activation, y_4_grad):
    x, w_s = build_tensors()
    new_y_2 = y_2_activation
    new_y_3 = w_s[2] @ new_y_2
    new_y_4 = w_s[3] @ new_y_3

    torch.autograd.backward(tensors=new_y_4, grad_tensors=y_4_grad)
    return x, w_s


def main():
    x, w_s, y_2_activation, y_4_grad = foward()
    x_, w_s_ = recompute(y_2_activation, y_4_grad)

    assert torch.equal(x, x_)
    for w, w_ in zip(w_s, w_s_):
        assert torch.equal(w, w_)

    w_s_with_grad = (2, 3)
    for i in range(len(w_s)):
        if i in w_s_with_grad:
            assert w_s[i].grad is not None
            assert w_s_[i].grad is not None
            assert torch.equal(w_s[i].grad, w_s_[i].grad)
        else:
            assert w_s[i].grad is not None
            assert w_s_[i].grad is None
    print("Activation checkpointing works!")


if __name__ == "__main__":
    main()
