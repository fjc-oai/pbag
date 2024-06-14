Effective Python The Book: Second Edition
By Brett Slatkin
https://effectivepython.com/



###################################################
# Chapter 1: pythonic thinking
###################################################
1. assignment expression
    if (x := foo()) == 3:
        bar(x)

###################################################
# Chapter 2: Lists and Dictionaries
###################################################
1. catch-all unpacking
    a, *b, c = [1, 2, 3, 4, 5]

2. sort using key parameter
    sorted(iterable, key=lambda x: x[1])

3. update nested dict
    d[key] = d.get(key, 0) + 1
    from collections import defaultdict
    d = defaultdict(lambda: defaultdict(list))


###################################################
# Chapter 3: Functions
###################################################
1. Closure variable parsing 
    a. reference a variable: 
        current function's scope > 
        any enclosing scope >
        global scope (module scope) >
        built-in scope >
        NameError
    b. assign a variable
        if already defined in the current scope:
            take the new value
        else:
            treat as a enw variable
    c. nonlocal to rescue

2. Starred expression
    a. starred expression: catch-all unpacking
        a, *b, c = a_list
    b. starred expression: define variable length argument function
        def foo(x, y, *z) # z as list
    c. starred expression: unpack iterables into function positional arguments
        foo(*a_list)
    d. doubled starred expression: define kwargs function
        def foo(x, y, **kwargs) # kwargs as dict
    e. double starred expression: unpack dict into function keyword arguments
        foo(**a_dict)

3. Positional-only and keyword-only arguments
    def foo(a, b, /, c, d, *, e, f)

4. Function decorator
    a. Use *args, **kwargs to pass through arguments. Use closure to access original function
        def wrapper(*args, **kwargs):
            results = func(*args, **kwargs)
    b. Calling decorator 
            @trace
            def fn()
        is essentially equavallent to
            fn = trace(fn)
    c. Use functions.wraps to preserve the function interface and metadata

# Chapter 5: classes 
    a. Use class as a stateful closure. __call__ method is a strong hint of being used a function argument.
    b. Use super().__init__() ensure MRO initialization order and diamond inheritance.
    c. MixIn: a very weird pattern in python
        extract out pluggable behaviors
        only define methods but no member variables
        methods will act on member variables from subclasses

# Chapter 6: metaclasses and attributes
    a. @property, @property.setter: associate customizations (e.g. computation, validation, etc) on getters and setters


