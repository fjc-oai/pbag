import torch
def test_1():
    print("Running test_1")
    torch.manual_seed(0)
    x = torch.rand(16, 8)
    x.requires_grad = True
    if x.requires_grad:
        x.register_hook(lambda grad: print(f"x.grad: {grad}"))
    y = torch.rand(8, 8)
    z = torch.mm(x, y)
    loss = z.sum()
    loss.backward()
    print("Done running test_1")

def test_2():
    print("Running test_2")
    torch.manual_seed(0)
    x = torch.rand(16, 8)
    x.requires_grad = True
    if x.requires_grad:
        x.register_hook(lambda grad: print(f"x.grad: {grad}"))
    x1 = x[:8]
    x2 = x[8:]
    if x1.requires_grad:
        x1.register_hook(lambda grad: print(f"x1.grad: {grad}"))
    if x2.requires_grad:
        x2.register_hook(lambda grad: print(f"x2.grad: {grad}"))
    y = torch.rand(8, 8)
    z1 = torch.mm(x1, y)
    z2 = torch.mm(x2, y)
    z = torch.cat([z1, z2], dim=0)
    loss = z.sum()
    loss.backward()
    print("Done running test_2")

def main():
    test_1()
    test_2()

if __name__ == "__main__":
    main()
