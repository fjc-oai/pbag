import torch


class _SetGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, target, grad):
        ctx.shape = target.shape
        ctx.grad = grad
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, torch.full(ctx.shape, ctx.grad), None
    

class Model(torch.nn.Module):
    def __init__(self, set_grad=True):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 1)
        self.set_grad = set_grad
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        
    def forward(self, epoch, x):
        x = self.fc1(x)
        x = self.fc2(x)
        if self.set_grad:
            x = _SetGrad.apply(x, self.dummy, epoch + 1)
        return x
    
def train_model(set_grad):
    torch.manual_seed(7)
    model = Model(set_grad)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = torch.nn.MSELoss()
    n_epochs = 3
    for epoch in range(n_epochs):
        opt.zero_grad()
        x = torch.rand(4, 10)
        target = torch.rand(4, 1)
        y = model(epoch, x)
        loss = loss_fn(y, target)
        loss.backward()
        opt.step()
        if set_grad:
            print(f"In epoch {epoch}: {model.dummy.grad=}")
    return model

def main():
    m1 = train_model(False)
    m2 = train_model(True)
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2
        if n1 == "dummy":
            print(f"{n1=} {p1} vs {p2}")
            assert torch.equal(p1, torch.tensor([0.0]))
            expected = sum([i + 1 for i in range(3)]) * 0.1 * -1
            assert torch.equal(p2, torch.tensor([expected])), f"{p2=}"
        else:
            assert torch.equal(p1, p2)
            print(f"Comparing {n1}: {torch.equal(p1, p2)=}")
    print("test pass")

if __name__ == "__main__":
    main()




