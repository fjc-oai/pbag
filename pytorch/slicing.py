import torch


def test_slicing():
    t = torch.arange(60).reshape(3, 4, 5)
    assert t.shape == (3, 4, 5)

    # Basic indexing
    assert t[0, 0, 0] == 0
    assert t[1, 0, 0] == 20

    # Basic slicing
    x = t[0, 0, 1:3]
    assert torch.equal(x, torch.tensor([1, 2]))

    # Multi-dimensional slicing
    x = t[0, 0:2, 1:3]
    assert torch.equal(x, torch.tensor([[1, 2], [6, 7]]))

    # Combined indexing and slicing
    x = t[0, 0:2, [1, 3]]
    assert torch.equal(x, torch.tensor([[1, 3], [6, 8]]))

    # Boolean indexing
    mask = t > 15
    x = t[mask]
    assert torch.equal(x, torch.arange(16, 60))

    mask = torch.tensor(
        [[True, False, False, False], [False, True, False, False], [False, False, True, False]]
    )
    x = t[mask]
    assert torch.equal(
        x, torch.tensor([[0, 1, 2, 3, 4], [25, 26, 27, 28, 29], [50, 51, 52, 53, 54]])
    )

    # In-place modification
    t = torch.arange(6).reshape(2, 3)
    mask = t % 2 == 0
    t[mask] = t[mask] + 1
    assert torch.equal(t, torch.tensor([[1, 1, 3], [3, 5, 5]]))


def main():
    test_slicing()
    print("slicing.py: All tests pass")


if __name__ == "__main__":
    main()
