"""

* tensor.view(dtype)
    t.view(torch.float16)
    doesn’t change underlying data. only change how to view the underlying data
    no intermediate tensor created

* tensor.to(dtype)
    x = t.to(torch.float16)
    type conversion happens under the hood, and data actually changed
    no in-place implementation 
    an intermediate tensor is created, and then assigned to x

* tensor.copy_()
    x.copy_(t)
    type conversion happens under the hood, thus data actually changed
    kind of a hacky implementation of in-place conversion, if an extra buffer has already been created

"""
import torch


def test_view():
    x = torch.randn(3,4, dtype=torch.float16)
    y = x.view(torch.bfloat16)
    assert x.dtype == torch.float16
    assert y.dtype == torch.bfloat16
    assert not (x-y < 1e-2).all()
    print("Values are different")

    x_bytes = x.numpy().tobytes()
    y_bytes = y.view(torch.float16).numpy().tobytes()
    assert x_bytes == y_bytes
    print("Bytes are same")

def test_to():
    x = torch.randn(3,4, dtype=torch.float16)
    y = x.to(torch.bfloat16)
    assert x.dtype == torch.float16
    assert y.dtype == torch.bfloat16
    assert (x-y < 1e-2).all()
    print("Values are same")

    x_bytes = x.numpy().tobytes()
    y_bytes = y.view(torch.float16).numpy().tobytes()
    assert x_bytes != y_bytes
    print("Bytes are different")

def test_copy_():
    x = torch.randn(3,4, dtype=torch.float16)
    y = torch.empty_like(x, dtype=torch.bfloat16)
    y.copy_(x)
    assert x.dtype == torch.float16
    assert y.dtype == torch.bfloat16
    assert (x-y < 1e-2).all()
    print("Values are same")

    x_bytes = x.numpy().tobytes()
    y_bytes = y.view(torch.float16).numpy().tobytes()
    assert x_bytes != y_bytes
    print("Bytes are different")

test_view()
test_to()
test_copy_()
print("All tests pass")