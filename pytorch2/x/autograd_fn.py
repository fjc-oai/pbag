import torch


def test_basic():
    x_grad_id = None
    x_grad_ptr = None
    y_grad_id = None
    y_grad_ptr = None

    class MyFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, y):
            ctx.save_for_backward(x, y)
            return x + y

        @staticmethod
        def backward(ctx, grad_output):
            x, y = ctx.saved_tensors
            x_grad = torch.full_like(x, 1.0)
            y_grad = torch.full_like(y, 2.0)
            nonlocal x_grad_id, y_grad_id, x_grad_ptr, y_grad_ptr
            x_grad_id = id(x_grad)
            y_grad_id = id(y_grad)
            x_grad_ptr = x_grad.storage().data_ptr()
            y_grad_ptr = y_grad.storage().data_ptr()
            return x_grad, y_grad

    x = torch.randn(4, 4, requires_grad=True)
    y = torch.randn(4, 4, requires_grad=True)
    z = MyFn.apply(x, y)
    z.sum().backward()

    assert x_grad_id != id(x.grad)
    assert y_grad_id != id(y.grad)
    assert x_grad_ptr == x.grad.storage().data_ptr()
    assert y_grad_ptr == y.grad.storage().data_ptr()
    print("test_basic passed")


def test_copy():
    x = torch.randn(4, 4, requires_grad=True)
    y = torch.randn(4, 4, requires_grad=True)
    x_grad = torch.full_like(x, 1.0)
    y_grad = torch.full_like(y, 2.0)

    class MyFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, y):
            ctx.save_for_backward(x, y)
            return x + y

        @staticmethod
        def backward(ctx, grad_output):
            x, y = ctx.saved_tensors
            return x_grad, y_grad

    z = MyFn.apply(x, y)
    z.sum().backward()

    assert id(x.grad) != id(x_grad)
    assert id(y.grad) != id(y_grad)
    assert x.grad.storage().data_ptr() != x_grad.storage().data_ptr()
    assert y.grad.storage().data_ptr() != y_grad.storage().data_ptr()

    print("test_copy passed")


def test_accumulate():
    x = torch.randn(4, 4, requires_grad=True)
    x.grad = torch.full_like(x, 1.0)
    y = torch.randn(4, 4, requires_grad=True)
    y.grad = torch.full_like(y, 1.0)
    x_grad_id = None
    y_grad_id = None
    x_grad_ptr = None
    y_grad_ptr = None

    class MyFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, y):
            ctx.save_for_backward(x, y)
            return x + y

        @staticmethod
        def backward(ctx, grad_output):
            x, y = ctx.saved_tensors
            x_grad = torch.full_like(x, 1.0)
            y_grad = torch.full_like(y, 2.0)
            nonlocal x_grad_id, y_grad_id, x_grad_ptr, y_grad_ptr
            x_grad_id = id(x_grad)
            y_grad_id = id(y_grad)
            x_grad_ptr = x_grad.storage().data_ptr()
            y_grad_ptr = y_grad.storage().data_ptr()
            return x_grad, y_grad

    z = MyFn.apply(x, y)
    z.sum().backward()

    assert x_grad_id != id(x.grad)
    assert y_grad_id != id(y.grad)
    assert x_grad_ptr != x.grad.storage().data_ptr()
    assert y_grad_ptr != y.grad.storage().data_ptr()
    assert torch.equal(x.grad, torch.full_like(x, 2.0))
    assert torch.equal(y.grad, torch.full_like(y, 3.0))

    print("test_accumulate passed")


def test_requires_grad_false():
    class MyFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, y):
            ctx.save_for_backward(x, y)
            return x + y

        @staticmethod
        def backward(ctx, grad_output):
            x, y = ctx.saved_tensors
            x_grad = torch.full_like(x, 1.0)
            y_grad = torch.full_like(y, 2.0)
            return x_grad, y_grad

    x = torch.randn(4, 4, requires_grad=False)
    y = torch.randn(4, 4, requires_grad=True)
    z = x + y
    z.sum().backward()
    assert x.grad is None
    assert y.grad is not None


def test_propagation():
    x_grad_id = None
    x_grad_ptr = None

    class MyFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, y):
            ctx.save_for_backward(x, y)
            return x + y

        @staticmethod
        def backward(ctx, grad_output):
            x, y = ctx.saved_tensors
            x_grad = torch.full_like(x, 1.0)
            nonlocal x_grad_id, x_grad_ptr
            x_grad_id = id(x_grad)
            x_grad_ptr = x_grad.storage().data_ptr()
            y_grad = torch.full_like(y, 2.0)
            return x_grad, y_grad

    x = torch.randn(4, 4, requires_grad=True)

    def hook_fn(grad):
        print("Running hook and examining grad")
        nonlocal x_grad_id, x_grad_ptr
        assert x_grad_id == id(grad)
        assert x_grad_ptr == grad.storage().data_ptr()

    x.register_hook(hook_fn)
    y = torch.randn(4, 4, requires_grad=True)
    x_ = x + 1
    z = MyFn.apply(x_, y)
    z.sum().backward()
    print("test_propagation passed")


def main():
    test_basic()
    test_copy()
    test_accumulate()
    test_requires_grad_false()
    test_propagation()


if __name__ == "__main__":
    main()
