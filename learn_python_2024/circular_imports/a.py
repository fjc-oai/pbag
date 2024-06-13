from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from b import B


class A:
    def __init__(self):
        pass

    def foo(self, b: 'B'):
        return b.fn()
    
    def fn(self):
        return 10
    