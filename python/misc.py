from typing import Any


def test_gc():
    import gc

    import torch

    t = torch.rand(3, 4)
    a = 123
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            gc_t = obj
    assert gc_t is t


def test_obj():
    class Foo:
        def __init__(self, x):
            self.x = x

    foo = Foo(123)
    foo.__extra__ = 456
    assert foo.__dict__["x"] == foo.x
    assert foo.__dict__["__extra__"] == foo.__extra__

    assert getattr(foo, "x") == foo.x
    assert getattr(foo, "__extra__") == foo.__extra__


def test_1():
    from collections import defaultdict

    m = defaultdict(list)
    m["a"].append(1)
    m["a"].append(2)
    m["b"].append(3)
    m2 = m.copy()
    m2["a"].pop()
    print(f"{m=}, {m2=}")
    import copy

    m3 = copy.deepcopy(m)
    m3["a"].pop()
    print(f"{m=}, {m3=}")

def f1():
    f2()

def f2():
    f3()

def f3():
    import traceback
    s=traceback.format_stack()
    print("\n".join(s))

def main():
    f1()


if __name__ == "__main__":
    main()
