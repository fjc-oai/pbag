from a import A


class B:
    def __init__(self):
        pass

    def foo(self, a: A):
        return a.fn()
    
    def fn(self):
        return 11