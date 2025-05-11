import torch


class Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        x_grad = torch.full_like(x, 0.5)
        return x_grad


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((2, 2), 10.0))

    def forward(self):
        return Fn.apply(self.weight)


def test_built_in_SGD():
    torch.manual_seed(42)

    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    print("*" * 100)
    print("built-in SGD")
    for i in range(3):
        optimizer.zero_grad()
        y = model()
        y.sum().backward()
        optimizer.step()
        print(f"step {i}")
        print(model.weight)


def test_tiny_SGD():
    torch.manual_seed(42)

    class TinySGD(torch.optim.Optimizer):
        def __init__(self, params, lr=0.1):
            super().__init__(params, defaults=dict(lr=lr))

        def step(self):
            for group in self.param_groups:
                for p in group["params"]:
                    p.data.sub_(group["lr"] * p.grad)

    model = Model()
    optimizer = TinySGD(model.parameters(), lr=0.1)
    print("*" * 100)
    print("tiny SGD")
    for i in range(3):
        optimizer.zero_grad()
        y = model()
        y.sum().backward()
        optimizer.step()
        print(f"step {i}")
        print(model.weight)


def test_built_in_Adam():
    torch.manual_seed(42)
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))
    print("*" * 100)
    print("built-in Adam")
    for i in range(3):
        optimizer.zero_grad()
        y = model()
        y.sum().backward()
        optimizer.step()
        print(f"step {i}")
        print(model.weight)


def test_tiny_Adam():
    torch.manual_seed(42)

    class TinyAdam(torch.optim.Optimizer):
        def __init__(self, params, lr=0.1, betas=(0.9, 0.999), eps=1e-8):
            super().__init__(params, defaults=dict(lr=lr, betas=betas, eps=eps))

        def step(self):
            for group in self.param_groups:
                for p in group["params"]:
                    state = self.state[
                        p
                    ]  # Store Adam state in Optimizer.state to be checkpointable and recoverable
                    if "step" not in state:
                        assert "m" not in state
                        assert "v" not in state
                        state["step"] = 0
                        state["m"] = torch.zeros_like(p.data)
                        state["v"] = torch.zeros_like(p.data)

                    state["step"] += 1
                    m, v = state["m"], state["v"]
                    beta1, beta2 = group["betas"]
                    m.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                    m_hat = m / (1 - beta1 ** state["step"])
                    v_hat = v / (1 - beta2 ** state["step"])
                    p.data.sub_(group["lr"] * m_hat / (v_hat.sqrt() + group["eps"]))

    model = Model()
    optimizer = TinyAdam(model.parameters(), lr=0.1, betas=(0.9, 0.999))
    print("*" * 100)
    print("tiny Adam")
    for i in range(3):
        optimizer.zero_grad()
        y = model()
        y.sum().backward()
        optimizer.step()
        print(f"step {i}")
        print(model.weight)


def test_tiny_shampoo():
    torch.manual_seed(42)

    # Copied from https://github.com/jettify/pytorch-optimizer/blob/master/torch_optimizer/shampoo.py
    class Shampoo(torch.optim.Optimizer):
        r"""Implements Shampoo Optimizer Algorithm.

        It has been proposed in `Shampoo: Preconditioned Stochastic Tensor
        Optimization`__.

        Arguments:
            params: iterable of parameters to optimize or dicts defining
                parameter groups
            lr: learning rate (default: 1e-3)
            momentum: momentum factor (default: 0)
            weight_decay: weight decay (L2 penalty) (default: 0)
            epsilon: epsilon added to each mat_gbar_j for numerical stability
                (default: 1e-4)
            update_freq: update frequency to compute inverse (default: 1)

        Example:
            >>> import torch_optimizer as optim
            >>> optimizer = optim.Shampoo(model.parameters(), lr=0.01)
            >>> optimizer.zero_grad()
            >>> loss_fn(model(input), target).backward()
            >>> optimizer.step()

        __ https://arxiv.org/abs/1802.09568

        Note:
            Reference code: https://github.com/moskomule/shampoo.pytorch
        """

        def __init__(
            self,
            params,
            lr: float = 1e-1,
            momentum: float = 0.0,
            weight_decay: float = 0.0,
            epsilon: float = 1e-4,
        ):
            if lr <= 0.0:
                raise ValueError("Invalid learning rate: {}".format(lr))
            if momentum < 0.0:
                raise ValueError("Invalid momentum value: {}".format(momentum))
            if weight_decay < 0.0:
                raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
            if epsilon < 0.0:
                raise ValueError("Invalid momentum value: {}".format(momentum))

            defaults = dict(
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                epsilon=epsilon,
            )
            super(Shampoo, self).__init__(params, defaults)

        def step(self, closure=None):
            """Performs a single optimization step.

            Arguments:
                closure: A closure that reevaluates the model and returns the loss.
            """
            loss = None
            if closure is not None:
                loss = closure()

            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    order = grad.ndimension()
                    original_size = grad.size()
                    state = self.state[p]
                    momentum = group["momentum"]
                    weight_decay = group["weight_decay"]
                    if len(state) == 0:
                        state["step"] = 0
                        if momentum > 0:
                            state["momentum_buffer"] = grad.clone()
                        for dim_id, dim in enumerate(grad.size()):
                            # precondition matrices
                            state["precond_{}".format(dim_id)] = group[
                                "epsilon"
                            ] * torch.eye(dim, out=grad.new(dim, dim))
                            state["inv_precond_{dim_id}".format(dim_id=dim_id)] = grad.new(
                                dim, dim
                            ).zero_()

                    if momentum > 0:
                        grad.mul_(1 - momentum).add_(
                            state["momentum_buffer"], alpha=momentum
                        )

                    if weight_decay > 0:
                        grad.add_(p.data, alpha=group["weight_decay"])

                    # See Algorithm 2 for detail
                    for dim_id, dim in enumerate(grad.size()):
                        precond = state["precond_{}".format(dim_id)]
                        inv_precond = state["inv_precond_{}".format(dim_id)]

                        # mat_{dim_id}(grad)
                        grad = grad.transpose_(0, dim_id).contiguous()
                        transposed_size = grad.size()
                        grad = grad.view(dim, -1)

                        grad_t = grad.t()
                        precond.add_(grad @ grad_t)

                        if dim_id == order - 1:
                            # finally
                            grad = grad_t @ inv_precond
                            # grad: (-1, last_dim)
                            grad = grad.view(original_size)
                        else:
                            # if not final
                            grad = inv_precond @ grad
                            # grad (dim, -1)
                            grad = grad.view(transposed_size)

                    state["step"] += 1
                    state["momentum_buffer"] = grad
                    p.data.add_(grad, alpha=-group["lr"])

            return loss

    model = Model()
    optimizer = Shampoo(model.parameters(), lr=0.1)
    print("*" * 100)
    print("tiny Shampoo")
    for i in range(3):
        optimizer.zero_grad()
        y = model()
        y.sum().backward()
        optimizer.step()
        print(f"step {i}")
        print(model.weight)


def main():
    test_built_in_SGD()
    test_tiny_SGD()
    test_built_in_Adam()
    test_tiny_Adam()
    test_tiny_shampoo()


if __name__ == "__main__":
    main()
