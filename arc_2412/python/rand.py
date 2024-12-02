def test_rand():
    import random

    N_RNG = 2
    seeds = [1, 42]
    names = ["rng_1", "rng_42"]
    rng_list_map = {
        name: [random.Random(seed) for _ in range(N_RNG)] for name, seed in zip(names, seeds)
    }
    rng_random_map = {
        name: [rng.random() for rng in rng_list] for name, rng_list in rng_list_map.items()
    }
    rng_randint_map = {
        name: [rng.randint(0, 100) for rng in rng_list] for name, rng_list in rng_list_map.items()
    }
    for name, num_list in rng_random_map.items():
        assert all([num == num_list[0] for num in num_list])
    assert rng_randint_map["rng_1"][0] != rng_randint_map["rng_42"][0]
    for name, num_list in rng_randint_map.items():
        assert all([num == num_list[0] for num in num_list])
    assert rng_randint_map["rng_1"][0] != rng_randint_map["rng_42"][0]

    print("test_rand passed")


def test_tensor_rand():
    import torch

    torch.manual_seed(2)
    a = torch.rand(2, 3)
    b = torch.rand(3, 2)
    torch.manual_seed(2)
    a_ = torch.rand(2, 3)
    b_ = torch.rand(3, 2)
    assert torch.equal(a, a_)
    assert torch.equal(b, b_)
    torch.manual_seed(42)
    a_ = torch.rand(2, 3)
    b_ = torch.rand(3, 2)
    assert not torch.equal(a, a_)
    assert not torch.equal(b, b_)
    print("test_tensor_rand passed")


def main():
    test_rand()
    test_tensor_rand()


if __name__ == "__main__":
    main()
