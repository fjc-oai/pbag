import torch

def test_tensor_view_to_cpu():
    t = torch.rand(2, 1024, 1024, device='cuda')
    x = t[:1]
    y = t[1:]
    assert x.storage().data_ptr() == t.storage().data_ptr()

    a = x.to('cpu')
    b = y.to('cpu')
    assert a.storage().data_ptr() != x.storage().data_ptr()
    print('test_tensor_view_to_cpu passed')

def main():
    test_tensor_view_to_cpu()

if __name__ == '__main__':
    main()

