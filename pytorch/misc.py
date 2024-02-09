import torch


def test_storage():
    t = torch.rand(4, 8)
    t2 = t[2:, :]
    assert t.storage().data_ptr() == t2.storage().data_ptr()
    assert t.data_ptr() != t2.data_ptr()


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


def test_seed():
    torch.manual_seed(41)
    tensors = [torch.rand(4, 8) for _ in range(10)]
    torch.manual_seed(41)
    tensors2 = [torch.rand(4, 8) for _ in range(10)]
    for t1, t2 in zip(tensors, tensors2):
        assert t1.equal(t2)


def main():
    test_storage()
    test_parameter()
    test_seed()


if __name__ == "__main__":
    main()
