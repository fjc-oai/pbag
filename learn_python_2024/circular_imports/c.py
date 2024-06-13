from a import A
from b import B

if __name__ == '__main__':
    a = A()
    b = B()
    print(a.foo(b))
    print(b.foo(a))