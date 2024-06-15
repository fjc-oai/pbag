def test_dis():
    def foo(a, b):
        return a + b

    import dis

    dis.dis(foo)


def test_free_variable():
    def foo():
        x = 42

        def bar():
            return x

        return id(x), bar

    x_id1, bar1 = foo()
    x_id2, bar2 = foo()
    bar1_x_id = id(bar1())
    bar2_x_id = id(bar2())
    assert x_id1 == x_id2 == bar1_x_id == bar2_x_id


def test_nonlocal():
    def foo():
        x = 42

        def bar():
            nonlocal x
            x += 1
            return x

        return bar

    bar = foo()
    assert bar() == 43
    assert bar() == 44


def test_simple_decorator():
    import functools

    def print_name(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"Calling {func.__name__}")
            return func(*args, **kwargs)

        return wrapper

    @print_name
    def foo():
        pass

    foo()


def test_lru_cache():
    n_invocations = 0
    import functools

    @functools.lru_cache(maxsize=3)
    def foo(n):
        nonlocal n_invocations
        n_invocations += 1
        return n

    foo(1)
    assert n_invocations == 1
    foo(1)
    assert n_invocations == 1
    foo(2)
    assert n_invocations == 2
    foo(3)
    assert n_invocations == 3
    foo(3)
    assert n_invocations == 3
    foo(4)
    assert n_invocations == 4
    foo(1)
    assert n_invocations == 5


def test_my_lru_cache():
    import functools

    def my_lru_cache(maxsize: int | None = None):
        def decorator(func):
            cache = {}

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                key = args + tuple(kwargs.items())
                if key in cache:
                    return cache[key]
                result = func(*args, **kwargs)
                if maxsize is not None and len(cache) >= maxsize:
                    cache.pop(next(iter(cache)))
                cache[key] = result
                return result

            return wrapper

        return decorator

    n_invocations = 0

    @my_lru_cache(maxsize=3)
    def foo(n):
        nonlocal n_invocations
        n_invocations += 1
        return n

    foo(1)
    assert n_invocations == 1
    foo(1)
    assert n_invocations == 1
    foo(2)
    assert n_invocations == 2
    foo(3)
    assert n_invocations == 3
    foo(3)
    assert n_invocations == 3
    foo(4)
    assert n_invocations == 4
    foo(1)
    assert n_invocations == 5
