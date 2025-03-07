import cprofiler

cprofiler.set_profile()

def foo():
    return 1

def bar():
    return foo() + 1

def baz():
    return bar() + 1

baz()

cprofiler.unset_profile()
