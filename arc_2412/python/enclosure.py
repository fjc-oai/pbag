class FuncWrapper:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def test_capture():
    x = [1, 2, 3]

    def foo():
        return len(x)

    func_wrapper = FuncWrapper(foo)
    assert func_wrapper() == 3
    x.append(4)
    assert func_wrapper() == 4


def test_capture_immutable():
    x = 3

    def foo():
        return x

    func_wrapper = FuncWrapper(foo)
    assert func_wrapper() == 3
    x = 5
    assert func_wrapper() == 5


def test_partial():
    import functools

    x = [1, 2, 3]

    def foo(x):
        return len(x)

    foo_ = functools.partial(foo, x)
    func_wrapper = FuncWrapper(foo_)
    assert func_wrapper() == 3
    x.append(4)
    assert func_wrapper() == 4


def test_partial_immutable():
    import functools

    x = 3

    def foo(x):
        return 3

    foo_ = functools.partial(foo, x)
    func_wrapper = FuncWrapper(foo_)
    assert func_wrapper() == 3
    x = 5
    assert func_wrapper() == 3
