import torch


def inplace_mul(inplace: bool):
    torch.manual_seed(7)
    w1 = torch.randn(2, 3, 5, requires_grad=True)
    w2 = torch.randn(2, 5, 4, requires_grad=True)

    logits = w1 @ w2 # (2, 3, 4)
    scale = torch.randn(3)
    if inplace:
        logits.mul_(scale.reshape(1, 3, 1))
    else:
        logits = logits.mul(scale.reshape(1, 3, 1))

    w3 = torch.randn(2, 4, 2, requires_grad=True)
    w4 = logits @ w3 # (2, 3, 2)

    loss = w4.sum()
    loss.backward()

    return w1.grad, w2.grad, w3.grad

def cmp_grads():
    grad1, grad2, grad3 = inplace_mul(True)
    grad1_, grad2_, grad3_ = inplace_mul(False)
    print(f"allclose(grad1, grad1_): {torch.allclose(grad1, grad1_)}")
    print(f"allclose(grad2, grad2_): {torch.allclose(grad2, grad2_)}")
    print(f"allclose(grad3, grad3_): {torch.allclose(grad3, grad3_)}")

if __name__ == "__main__":
    cmp_grads()


