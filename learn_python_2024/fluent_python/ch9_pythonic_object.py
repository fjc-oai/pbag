import pytest


def test_slots():
    class Foo:
        __slots__ = ("x", "y")

        def __init__(self, x, y):
            self.x = x
            self.y = y

    class Bar:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    foo = Foo(1, 2)
    bar = Bar(1, 2)
    with pytest.raises(AttributeError):
        foo.z = 3
    bar.z = 3
