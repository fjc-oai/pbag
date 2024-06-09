"""
Item 21: Know How Closures Interact with Variable Scope
"""
def test_closures():
    def special_sort(values, special_values):
        def helper(x):
            if x in special_values:
                return (0, x)
            return (1, x)
        values.sort(key=helper)
    values = [3, 1, 2, 4, 10, 15]
    special_values = {4, 10}
    special_sort(values, special_values)   
    assert values == [4, 10, 1, 2, 3, 15]

    def special_sort_and_find_wrong(values, special_values):
        has_special_value = False
        def helper(x):
            if x in special_values:
                has_special_value = True
                return (0, x)
            return (1, x)
        values.sort(key=helper)
        return has_special_value
    
    def special_sort_and_find(values, special_values):
        has_special_value = False
        def helper(x):
            nonlocal has_special_value
            if x in special_values:
                has_special_value = True
                return (0, x)
            return (1, x)
        values.sort(key=helper)
        return has_special_value
    
    values = [3, 1, 2, 4, 10, 15]
    special_values = {4, 10}
    assert special_sort_and_find_wrong(values, special_values) == False
    assert special_sort_and_find(values, special_values) == True


"""
Item 22: Reduce Visual Noise with Variable Positional Arguments
"""
def test_variable_positional_arguments():
    # starred expression: catch-all unpacking
    l = [1,2,3,4,5]
    a, *b, c = l
    assert b == [2,3,4]

    # starred expression: unpack iterables into function arguments
    l = [1,2,3]
    def foo(a, b, c):
        return a + b + c
    assert foo(*l) == 6

    # starred expression: variable length function arguments
    def msum(*values):
        return sum(values)
    assert msum(1,2,3,4,5) == 15

    # double starred expression: unpack dict into function arguments
    d = {'a': 1, 'b': 2}
    def foo(a, b):
        return a + b
    assert foo(**d) == 3

    # double starred expression: kwargs in function arguments
    def foo(**kwargs):
        return kwargs
    assert foo(a=1, b=2) == {'a': 1, 'b': 2}

    # mixed starred expression: variable length function arguments and kwargs
    def foo(a, *b, **c):
        return a, b, c
    assert foo(1, 2, 3, x=4, y=5) == (1, (2, 3), {'x': 4, 'y': 5})

"""
Item 25: Enforce Clarity with Keyword-Only and Positional-Only Arguments
"""
def test_keyword_only_and_positional_only_arguments():
    def foo(a, b, /, *, d, e):
        return a, b, d, e
    assert foo(1, 2, d=3, e=4) == (1, 2, 3, 4)


"""
Item 26: Define Function Decorators with functools.wraps
"""
def test_function_decorators():
    def trace(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            print(f"{func.__name__}({args!r}, {kwargs!r}) -> {result!r}")
            return result
        return wrapper

    @trace
    def foo(a, b):
        """Return the sum of two numbers."""
        return a + b

    assert foo(1, 2) == 3
    assert foo.__name__ == "wrapper"
    assert foo.__doc__ == None

    import functools
    def better_trace(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            print(f"{func.__name__}({args!r}, {kwargs!r}) -> {result!r}")
            return result
        return wrapper

    @better_trace
    def foo(a, b):
        """Return the sum of two numbers."""
        return a + b
    assert foo(1, 2) == 3
    assert foo.__name__ == "foo"
    assert foo.__doc__ == "Return the sum of two numbers."