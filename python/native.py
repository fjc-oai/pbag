from typing import Any


class Foo:
    def __init__(self):
        self.x = 42
        self.y = 43

    def add(self):
        return self.x + self.y

    def __getattr__(self, name):
        print(f"calling __getattr__({name})")
        if name.startswith("call_"):
            name = name[5:]
            assert hasattr(self, name)
            return getattr(self, name)


def test_getattr():
    f = Foo()
    assert f.add() == f.call_add()


def main():
    test_getattr()


if __name__ == "__main__":
    main()
