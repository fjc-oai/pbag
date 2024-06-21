"""
Type hints cheat sheet: https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html

Generic types
    - Used in class/method definitions: so that all of the methods/attributes marked with the same type variable are of the same type.
        T = TypeVar('T')
        class Box(Generic[T]):
            def foo(self, x: T) -> T:
    - Used in derived class definitation: so that the derived class can use the same type variable as the base class.
        class AnotherBox(Box[T]):
    - Used in object type hinting: so that the object is of the same type as the type variable.
        box: Box[int] = Box(10)
        def foo(box: Box[int]) -> Box[str]:
    - Type(int) is of type type

Type hints
    - Type hints are used by mypy for type checking. No runtime effect.
    - Use reveal_type(x) to see the type of x.
    - cast can enforce a type to override type inferred by mypy. No runtime effect.
    - type() vs reveal_type(): actual type vs mypy inferred type.

forward reference & TYPE_CHECKING
    - Use quotes to refer to a class that is defined later in the file.
    - Use TYPE_CHECKING to avoid circular imports.

Protocol
    - Two types of type checking: nominal and structural subtyping
    - Nominal subtyping: class A is a subtype of class B if A is derived from B.
    - Structural subtyping: class A is a subtype of class B if A has all the methods/attributes of B.
    - Protocol is used for structural subtyping.
    - Protocol can also be templated with TypeVar.
"""

def test_generic_types() -> None:
    from typing import Generic, TypeVar
    T = TypeVar('T')
    class Box(Generic[T]):
        def __init__(self, x: T) -> None:
            self.x = x

        def foo(self) -> T:
            return self.x
    
    box = Box(10)
    assert box.foo() == 10
    box2 = Box('hello')
    assert box2.foo() == 'hello'
    # z = box.foo() + box2.foo() 
    # ^^^ expected mypy error ^^^

    def foo(a: T) -> T:
        return a
    assert foo(10) == 10
    # foo(10) + foo('hello')
    # ^^^ expected mypy error ^^^

    class AnotherBox(Box[T]):
        def __init__(self, x: T) -> None:
            super().__init__(x)
        
        def bar(self) -> T:
            return self.x
    anotherbox = AnotherBox(20)
    assert anotherbox.foo() == 20


def test_type_hint() -> None:
    from typing import cast
    a = 10 
    b = 'hello'
    # a + b 
    # ^^^ expected mypy error ^^^
    c = cast(int, b)
    # a + c
    # ^^^ no mypy error ^^^
    # but both of the above will raise runtime error


def test_forward_reference() -> None:
    class A:
        def foo(self, b: 'B') -> int:
            return b.bar()
    
    class B:
        def bar(self) -> int:
            return 10
        

def test_protocol() -> None:
    from typing import Protocol, TypeVar
    T = TypeVar('T')
    class Fooable(Protocol[T]):
        def foo(self, t: T) -> T:
            ...
    
    class A:
        def foo(self, x: int) -> int:
            return x
    
    class B:
        def foo(self, x: str) -> str:
            return x
        
    class C:
        def foo(self, x: float) -> str:
            return str(x)
        
    def call_foo(obj: Fooable[T], x: T) -> T:
        return obj.foo(x)
    
    a = A()
    b = B()
    c = C()
    assert call_foo(a, 10) == 10
    assert call_foo(b, 'hello') == 'hello'
    # assert call_foo(c, 3.14) == '3.14'
    # ^^^ expected mypy error ^^^