import torch

bs = 8
n_ctx = 256
d_model = 128

q = torch.randn(bs, n_ctx, d_model).cuda()  # [8, 256, 128] -> 1MB
q.requires_grad = True
k = torch.randn(bs, n_ctx, d_model).cuda()
k.requires_grad = True
v = torch.randn(bs, n_ctx, d_model).cuda().requires_grad_()
v.requires_grad = True

x = q @ k.transpose(-2, -1)  # [8, 256, 256] -> 2MB
x = torch.softmax(x, dim=-1)  # 2MB
x = x @ v  # 8*256*128*4=1MB

grad = torch.randn_like(x)
x.backward(grad)

grad = torch.randn_like(grad)

torch.cuda.memory._record_memory_history()
x = k.transpose(-2, -1)  # 1MB
x = q @ x # [8, 256, 256] -> 2MB
x = torch.softmax(x, dim=-1)  # 2MB
x = x @ v  # 8*256*128*4=1MB
x.backward(grad)
torch.cuda.memory._dump_snapshot(f"peak.pickle")
