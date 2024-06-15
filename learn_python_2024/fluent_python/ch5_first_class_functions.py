def test_dir_and_dict():
    class Foo:
        def __init__(self):
            self.x = 1
            self.y = 2

        def fn(self):
            return self.x + self.yß

    foo = Foo()
    foo.z = 3
    foo.__bar__ = 4

    print(f"dir(foo): {dir(foo)}")
    print(f"foo.__dict__: {foo.__dict__}")


def test_positional_and_keyword_parameters():
    def foo(a, b, *, c, d, **kwargs):
        print(f"a: {a}, b: {b}, c: {c}, d: {d}, kwargs: {kwargs}")

    foo(1, 2, c=3, d=4, e=5, f=6)


def test_inspection():
    def foo(a: int, b: int) -> int:
        """this is a add function."""
        return a + b

    print(f"foo.__name__: {foo.__name__}")
    print(f"foo.__doc__: {foo.__doc__}")
    print(f"foo.__annotations__: {foo.__annotations__}")
    print(f"foo.__defaults__: {foo.__defaults__}")
    print(f"foo.__code__.co_varnames: {foo.__code__.co_varnames}")
    print(f"foo.__code__.co_argcount: {foo.__code__.co_argcount}")


def test_partial():
    from functools import partial

    def foo(a, b, c, d):
        return a + b + c + d

    bar = partial(foo, 1, 2, 3)
    assert bar(4) == 10


def test_my_partial():
    class MyPartial:
        def __init__(self, func, *args, **kwargs):
            self.func = func
            self.args = args
            self.kwargs = kwargs

        def __call__(self, *args, **kwargs):
            return self.func(*(self.args + args), **{**self.kwargs, **kwargs})

    def foo(a, b, c, d):
        return a + b + c + d

    bar = MyPartial(foo, 1, 2, 3)
    assert bar(4) == 10
