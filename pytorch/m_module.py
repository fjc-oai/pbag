import torch


def test_parameter():
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.w = torch.nn.Parameter(torch.rand(4, 8))
            self.b = torch.nn.Parameter(torch.rand(4, 8))
            self.l = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.rand(4, 8)) for _ in range(3)]
            )

    model = Model()
    assert model.w.data.shape == (4, 8)
    params = list(model.parameters())
    assert len(params) == 5
