import torch

import numpy as np

w_data = np.random.rand(3,5)
x_data = np.random.rand(8,3)
bs = x_data.shape[0]

x1, x2, x = [torch.tensor(arr) for arr in [x_data[:bs//2,:], x_data[bs//2:,:], x_data]]
w1, w2, w = [torch.tensor(arr, requires_grad=True) for arr in [w_data] * 3]

l1 = torch.mean(x1 @ w1)
l1.backward()
l2 = torch.mean(x2 @ w2)
l2.backward()
l3 = torch.mean(x @ w)
l3.backward()

assert torch.allclose((w1.grad + w2.grad)/2, w.grad)
print('All passed')

