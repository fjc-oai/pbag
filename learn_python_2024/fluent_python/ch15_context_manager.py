import contextlib

import pytest

global_var = 0


def set_global_var(val):
    global global_var
    global_var = val


def get_global_var():
    return global_var


class FixedGlobalVar:
    def __init__(self, val):
        self.val = val
        self.old_val = None

    def __enter__(self):
        self.old_val = get_global_var()
        set_global_var(self.val)

    def __exit__(self, exc_type, exc_value, traceback):
        set_global_var(self.old_val)


def test_context_manager():
    assert get_global_var() == 0
    with FixedGlobalVar(1):
        assert get_global_var() == 1
    assert get_global_var() == 0


@contextlib.contextmanager
def fixed_global_var(val):
    old_val = get_global_var()
    set_global_var(val)
    yield
    print("im executing the yield statement")
    set_global_var(old_val)


def test_context_manager_decorator():
    assert get_global_var() == 0
    with fixed_global_var(1):
        assert get_global_var() == 1
    assert get_global_var() == 0


def test_context_manager_exception():
    assert get_global_var() == 0
    with pytest.raises(ValueError):
        with fixed_global_var(1):
            assert get_global_var() == 1
            raise ValueError()
    assert (
        get_global_var() == 1
    )  # context manager failed to recover the old value when exception is raised


@contextlib.contextmanager
def better_fixed_global_var(val):
    old_val = get_global_var()
    set_global_var(val)
    try:
        yield
    finally:
        set_global_var(old_val)


def test_better_context_manager_decorator():
    set_global_var(0)
    assert get_global_var() == 0
    with pytest.raises(ValueError):
        with better_fixed_global_var(1):
            assert get_global_var() == 1
            raise ValueError()
    assert get_global_var() == 0
