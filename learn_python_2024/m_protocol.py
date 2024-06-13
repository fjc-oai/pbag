from typing import Protocol, TypeVar

T = TypeVar('T')

class Fooable(Protocol):
    def foo(self, t: T) -> T:
        ...

class A:
    def __init__(self):
        pass

    def foo(self, x: int):
        return x

class B:
    def __init__(self):
        pass

    def foo(self, x: str):
        return x
    
class C:
    def __init__(self):
        pass

    def foo(self, x: float) -> str:
        return str(x)
    
def call_foo(obj: Fooable, x: T) -> T:
    return obj.foo(x)

if __name__ == '__main__':
    a = A()
    b = B()
    c = C()
    print(call_foo(a, 10))
    print(call_foo(b, 'hello'))
    print(call_foo(c, 3.14))