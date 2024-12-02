import functools

import lib


def wrapper(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)

    return wrapped


def apply():
    methods = ["foo", "bar"]
    for k, v in lib.Client.__dict__.items():
        if callable(v) and k in methods:
            setattr(lib.Client, k, wrapper(v))


import sys

imported_lib = sys.modules["lib"]
print(f"Before patching: {imported_lib.Client.foo=}")
apply()
print(f"After patching: {imported_lib.Client.foo=}")
