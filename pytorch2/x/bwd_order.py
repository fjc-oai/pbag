import torch

x = torch.randn(1024, 16, requires_grad=True, device="cuda")
y = torch.randn(16, 16, requires_grad=True, device="cuda")

res = []

for i, idx in enumerate(range(0, 1024, 128)):
    x_sub = x[idx:idx+128]
    
    # Register hook to print when gradient for this chunk is computed
    x_sub.register_hook(lambda grad, i=i: print(f"Backward for x_sub_{i}, grad norm: {grad.norm()}"))

    out = x_sub @ y
    res.append(out)

out = torch.cat(res, dim=0)
loss = out.pow(2).mean()
loss.backward()