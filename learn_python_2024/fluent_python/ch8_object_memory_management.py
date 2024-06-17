import copy


def test_identity_and_equality():
    l = [1, 2, 3]
    l2 = l
    l3 = list(l)
    assert l2 is l
    assert l2 == l
    assert l3 is not l
    assert l3 == l


def test_shallow_copy_list():
    l = [1, 2, [3, 4]]
    l2 = copy.copy(l)
    assert l2 is not l
    assert l2 == l

    l.append(5)
    assert 5 not in l2
    l[2].append(6)
    assert 6 in l2[2]


def test_shallow_copy_dict():
    d = {"a": 1, "b": [2, 3]}
    d2 = copy.copy(d)
    assert d2 is not d
    assert d2 == d

    d["c"] = 3
    assert "c" not in d2
    d["b"].append(4)
    assert 4 in d2["b"]


def test_shallow_copy_object():
    class A:
        def __init__(self):
            self.x = [1, 2, 3]

    a = A()
    a2 = copy.copy(a)
    assert a2 is not a
    assert a2.x == a.x

    a.x.append(4)
    assert 4 in a2.x


def test_deep_copy_list():
    l = [1, 2, [3, 4]]
    l2 = copy.deepcopy(l)
    assert l2 is not l
    assert l2 == l

    l.append(5)
    assert 5 not in l2
    l[2].append(6)
    assert 6 not in l2[2]


def test_deep_copy_object():
    class A:
        def __init__(self):
            self.x = [1, 2, 3]

    a = A()
    a2 = copy.deepcopy(a)
    assert a2 is not a
    assert a2.x == a.x

    a.x.append(4)
    assert 4 not in a2.x

def test_mutable_as_default_param():
    def foo(a, b=[]):
        b.append(a)
        return b

    assert foo(1) == [1]
    assert foo(2) == [1, 2]
    assert foo(3) == [1, 2, 3]

def test_mutable_arguement():
    class Foo:
        def __init__(self, x):
            self.x = x
        def add(self, a):
            self.x.append(a)
    
    class Bar:
        def __init__(self, x):
            self.x = list(x)
        def add(self, a):
            self.x.append(a)

    l1 = [1, 2, 3]
    f = Foo(l1)
    f.add(4)
    assert l1 == [1, 2, 3, 4]

    l2 = [1, 2, 3]
    b = Bar(l2)
    b.add(4)
    assert l2 == [1, 2, 3]
