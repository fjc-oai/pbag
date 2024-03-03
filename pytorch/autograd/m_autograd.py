# A simple implementation of PyTorch Tensor AutoGrad

import torch
import numpy as np

class Tensor:
    def __init__(self, data: np.ndarray, requires_grad: bool =False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None

    def backward(self, grad=None):
        if self.requires_grad:
            if grad is None:
                grad = np.array([1.0])
            if self.grad is None:
                self.grad = grad
            else:
                self.grad.data += grad.data
            if self.grad_fn:
                self.grad_fn.backward(grad)

    def __matmul__(self, other: 'Tensor'):
        requires_grad = any([self.requires_grad, other.requires_grad])
        result = Tensor(self.data @ other.data, requires_grad=requires_grad)
        result.grad_fn = MatMulGradient(self, other)
        return result

    def __add__(self, other: 'Tensor'):
        requires_grad = any([self.requires_grad, other.requires_grad])
        result = Tensor(self.data + other.data, requires_grad=requires_grad)
        result.grad_fn = AddGradient(self, other)
        return result

    def mean(self):
        result = Tensor(np.mean(self.data), requires_grad=self.requires_grad)
        result.grad_fn = MeanGradient(self)
        return result

    def relu(self):
        result = Tensor(np.where(self.data>0,self.data,0), requires_grad=self.requires_grad)
        result.grad_fn = ReluGradient(self)
        return result

    def __str__(self):
        return f'{self.data}'


class MatMulGradient:
    def __init__(self, x: Tensor, y: Tensor):
        self.x = x
        self.y = y

    def backward(self, grad: np.ndarray):
        x_grad = grad @ self.y.data.T
        y_grad = self.x.data.T @ grad
        self.x.backward(x_grad)
        self.y.backward(y_grad)

class AddGradient:
    def __init__(self, x: Tensor, y: Tensor):
        self.x = x
        self.y = y

    def backward(self, grad: np.ndarray):
        self.x.backward(grad)
        self.y.backward(grad)

class MeanGradient:
    def __init__(self, x: Tensor):
        self.x = x

    def backward(self, grad: np.ndarray):
        x_grad = np.ones_like(self.x.data) / len(self.x.data)
        self.x.backward(x_grad)

class ReluGradient:
    def __init__(self, x: Tensor):
        self.x = x
    def backward(self, grad: np.ndarray):
        x_grad = np.where(self.x.data>0, 1, 0) * grad
        self.x.backward(x_grad)
        
def test():
    x_data=np.random.rand(4,6)
    w1_data=np.random.rand(6,8)
    w2_data=np.random.rand(8,1)
    x=Tensor(x_data)
    w1=Tensor(w1_data,requires_grad=True)
    w2=Tensor(w2_data,requires_grad=True)

    z1 = x @ w1
    a1 = z1.relu()
    z2 = a1 @ w2
    y = z2.mean()

    y.backward()

    x_t=torch.tensor(x_data)
    w1_t=torch.tensor(w1_data, requires_grad=True)
    w2_t=torch.tensor(w2_data, requires_grad=True)

    y_t = torch.mean(torch.relu(x_t @ w1_t) @ w2_t)

    y_t.backward()

    assert np.allclose(y.data, y_t.detach().numpy())
    assert np.allclose(w1.grad, w1_t.grad)
    assert np.allclose(w2.grad, w2_t.grad)
    print('All passed')
    
if __name__ == '__main__':
    test()