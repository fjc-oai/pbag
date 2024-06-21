import functools
import types
from collections import namedtuple


class Base:
    base_class_attr: int = 1

    def __init__(self):
        self.base_attr = 2

    def base_fn(self):
        return 3
    
class Derived(Base):
    derived_class_attr: int = 4

    def __init__(self):
        super().__init__()
        self.derived_attr = 5

    def derived_fn(self):
        return 6

class Foo(Derived):
    foo_class_attr: int = 7

    def __init__(self):
        super().__init__()
        self.foo_attr = 8

    def foo_fn(self):
        return 9
    

foo1 = Foo()
foo2 = Foo()
# breakpoint()

"""
TODO:


https://docs.python.org/3/howto/index.html
- scoket programming
- urllib
- ipaddress
- asyncio/event-loop/await

"""

"""
Attributes:
    - Data attributes
        - Instance attributes
        - Class attributes
    - Method attributes
        - Instance methods
        - Class methods
        - Static methods

"""
# object.__dict__ stores the instance data attributes, both defined and inherited
# object.__dict__ doesn't include its instance methods
assert foo1.__dict__ == {'base_attr': 2, 'derived_attr': 5, 'foo_attr': 8}

# object.__class__ stores class data attributes, instance methods, class methods, static methods
assert "foo_class_attr" in foo1.__class__.__dict__
assert "foo_fn" in foo1.__class__.__dict__

# function vs bound method
print(Foo.foo_fn)  # <function Foo.foo_fn at 0x7f8b3b7b7d30>
print(foo1.foo_fn)  # <bound method Foo.foo_fn of <__main__.Foo object at 0x7f8b3b7b7a90>>

# bind a method
def foo_fn_new(instance):
    return instance.foo_fn() + 1
foo1.foo_fn_new = types.MethodType(foo_fn_new, foo1)
assert foo1.foo_fn_new() == foo1.foo_fn() + 1

# access base class by __bases__
# __bases__ doesn't appear in the __dict__ of a class. Likely it's a special attribute accessed by the interpreter
assert Foo.__bases__ == (Derived,)
assert foo1.__class__.__bases__ == (Derived,)


class Bar:
    def __init__(self):
        self.bar_attr = 10

    def bar_fn(self):
        return 11
    
    def __new__(cls):
        # The callstack to __new__ is bar = Bar() -> Bar.__new__(Bar) -> Bar.__init__(bar)
        # This implies __new__ is handled by the interpreter
        # 
        # Also, python uses two staged construction: __new__ and __init__
        # __init__() will only be called if __new__() returns an instance of the class.
        # 
        # Some usage of __new__:
        # - Singleton pattern
        # - Caching
        # - Factory pattern
        # - Customizing object creation
        # - Immutable objects
        # - Metaclasses
        return super().__new__(Foo)
    

bar = Bar()


"""
descriptor
- __get__(self, instance, owner)
- __set__
- __delete__
"""

"""
https://docs.python.org/3/howto/descriptor.html
"""

class Ten:
    def __get__(self, instance, objtype=None):
        return 10
class A:
    x = 5
    y = Ten()

a = A()


class Max:
    def __get__(self, obj, objtype=None):
        return max(obj.values)
    
class Ages:
    max_age = Max()
    def __init__(self, values):
        self.values = values

ages = Ages([10, 20, 30])


class ValidatedAge:
    def __get__(self, obj, objtype=None):
        return obj._age
    def __set__(self, obj, value):
        if not isinstance(value, int):
            raise ValueError("Age must be an integer")
        if value < 0 or value > 150:
            raise ValueError("Age must be between 0 and 150")
        obj._age = value

class Person:
    age = ValidatedAge() # age is a class attribute, but the underlying data attribute is _age, which is an instance attribute
    def __init__(self, age):
        self.age = age # Note: this will call ValidatedAge.__set__!!! and thus creates self._age attribute

p1 = Person(10)
p2 = Person(20)


class PositiveNum:
    def __set_name__(self, owner, name):
        # interpreter will call this method to set the name of the descriptor, from the variable name

        """
        Sometimes it is desirable for a descriptor to know what class variable name it was assigned to. When a new class is created, the type metaclass scans the dictionary of the new class. If any of the entries are descriptors and if they define __set_name__(), that method is called with two arguments. The owner is the class where the descriptor is used, and the name is the class variable the descriptor was assigned to.


        """
        self.private_name = f"_{name}"

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)
    
    def __set__(self, obj, value):
        if not isinstance(value, int):
            raise ValueError(f"{self.name} must be an integer")
        setattr(obj, self.private_name, value)

class Person:
    age = ValidatedNum()
    height = ValidatedNum()
    def __init__(self, age, height):
        self.age = age
        self.height = height

p1 = Person(10, 100)

"""
Instance lookup scans through a chain of namespaces giving data descriptors the highest priority, followed by instance variables, then non-data descriptors, then class variables, and lastly __getattr__() if it is provided.

"""



# - __getattr__, __getattribute__, __get__
# __getattr__ is called when an attribute is not found in the usual places (instance, class, and base classes)
# __getattribute__ is called for every attribute access. implement this method with caution as it can lead to infinite recursion
# __get__ is called when an attribute is accessed through a descriptor



"""
Metaprogramming
1. using type() to generate classes
2. use class decorators
3. use metaclasses

namedtuple
type(type) == type
"""
Student = namedtuple("Student", ["name", "age"])
s = Student("Alice", 20)
assert s.name == "Alice"
assert s.age == 20


assert type(s) == Student
assert type(Student) == type
assert type(type) == type

def class_factory(cls_name, field_names):

    field_names = list(field_names)

    def __init__(self, *args, **kwargs):
        attrs = dict(zip(field_names, args))
        attrs.update(kwargs)
        for name, value in attrs.items():
            setattr(self, name, value)

    def __repr__(self):
        return f"{cls_name}({', '.join([f'{name}={getattr(self, name)}' for name in field_names])})"
    
    cls_attrs = {
        "__init__": __init__,
        "__repr__": __repr__,
    }
    return type(cls_name, (object,), cls_attrs)

Person = class_factory("Person", ["name", "age"])
p = Person("Alice", 20, height=170)
Cat = class_factory("Cat", ["name", "age", "color"])
c = Cat("Tom", 2, "black", weight=5)



def log_decorator(func, cls):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if args and isinstance(args[0], cls):
        # if args and args[0].__class__.__name__ == cls:
            print(f"Calling member method {func.__name__} on {args[0]}")
        else:
            print(f"Calling static method {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

def log_all_methods_decorator(cls):
    for name, value in cls.__dict__.items():
        if callable(value):
            setattr(cls, name, log_decorator(value, cls))
    return cls

@log_all_methods_decorator
class MyClass:
    def __init__(self):
        pass
    def my_method(self):
        pass

    @staticmethod
    def my_static_method():
        pass

my_instance = MyClass()
my_instance.my_method()
MyClass.my_static_method()    


class LogMethodsMeta(type):
    def __new__(cls, cls_name, bases, cls_attrs):
        # for name, value in cls_attrs.items():
        #     if callable(value):
        #         cls_attrs[name] = log_decorator(value, cls_name)
        new_class = super().__new__(cls, cls_name, bases, cls_attrs)
        for name, value in cls_attrs.items():
            if callable(value):
                setattr(new_class, name, log_decorator(value, new_class))
        return new_class
    

class MyClass(metaclass=LogMethodsMeta):
    def __init__(self):
        pass
    def my_method(self):
        pass

    @staticmethod
    def my_static_method():
        pass

my_instance = MyClass()

my_instance.my_method()
MyClass.my_static_method()



"""
regex
https://docs.python.org/3/howto/regex.html#more-metacharacters
"""

import re

# simple patterns

p = re.compile(r"abc")
m = p.search("asdfabcasdf")
print(f"{m.start()}, {m.end()}, {m.group()}")


# metacharacters

#   character class
p = re.compile(r"[abc][123]")
m = p.search("aaabbb111222ccc")
print(f"{m.start()}, {m.end()}, {m.group()}")

#  dot^ complementing the set
p = re.compile(r"[^abc]")
m = p.search("abc1abc2abc3")
print(f"{m.start()}, {m.end()}, {m.group()}")

#   escape by /
#       /d: [0-9]
#       /D: [^0-9]
#       /s: [ \t\n\r\f\v]
#       /w: [a-zA-Z0-9_]


#   | : or
print("start or")
p = re.compile(r"h[e|a]llo")
for g in p.finditer("hello, hallo, hillo"):
    print(g.start(), g.end(), g.group())
print("end or")

p = re.compile(r"h(ello|alo) world")
for g in p.finditer("hello world and halo world"):
    print(g.start(), g.end(), g.group())
print("end or")

#  ^ : start of the string
#  $ : end of the string
#  \b: word boundary

# repetition
#   {m, n}: at least m, at most n
#   * == {0,}
#   + == {1,}
#   ? == {0,1}
p = re.compile(r"[0-9]{3}")
m = p.search("abc123def456")
print(f"{m.start()}, {m.end()}, {m.group()}")

# use r"" to avoid espacing

# performing matches
#   match(): match from the beginning
#   search(): search the entire string
#   findall(): find all matches
#   finditer(): find all matches as iterator

print("findall")
p = re.compile(r"[a-z]\d")
res = p.findall("abc123def456")
for match in res:
    print(match)


print("finditer")
p = re.compile(r"[a-z]\d")
res = p.finditer("abc123def456")
for match in res:
    print(match.start(), match.end(), match.group())

# grouping
print("grouping")
p = re.compile(r"\+?(\d)-(\d{3})-(\d{3})-(\d{4})")
itr = p.finditer("my phone number is 1-123-456-7890 and +2-234-567-8901")
for match in itr:
    print(f"{match.start()}, {match.end()}, {match.group(0)} {match.group(1)} {match.group(2)} {match.group(3)} {match.group(4)}")
print("end grouping")