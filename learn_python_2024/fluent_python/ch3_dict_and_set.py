from collections import Counter, OrderedDict, defaultdict


def test_hashable():
    class Foo:
        def __init__(self, name):
            self.name = name

    foo1 = Foo("foo")
    foo2 = Foo("foo")
    d = {foo1: 1, foo2: 2}
    print(f"d: {d}")
    assert len(d) == 2


def test_dict():
    d = defaultdict(lambda: defaultdict(list))
    d["a"]["b"].append(1)
    d["a"]["b"].append(2)
    assert d == {"a": {"b": [1, 2]}}


def test_ordered_dict():
    d = OrderedDict()
    d["a"] = 1
    d["c"] = 3
    d["b"] = 2
    assert list(d.keys()) == ["a", "c", "b"]  # reserve insertion order


def test_counter():
    c = Counter()
    c.update(["a", "b", "a", "c"])
    assert c.most_common(1) == [("a", 2)]


def test_set():
    s1 = {1, 2, 3}
    s2 = {2, 3, 4}
    assert s1 & s2 == {2, 3}
    assert s1 | s2 == {1, 2, 3, 4}
