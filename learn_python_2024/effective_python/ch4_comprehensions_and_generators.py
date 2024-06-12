"""
Item 29: Avoid repeating work in comprehensions by using assignment expressions
"""
def test_assignment_expression():
    d = {
        "a": [1, 2, 3],
        "b": [4, 5],
        "c": [6, 7, 8, 9],
    }
    s = {k: l for k, v in d.items() if (l := len(v)) > 2}
    assert s == {"a": 3, "c": 4}

"""
Item 30: Consider generator expressions for large comprehensions
"""
def test_generator_expression():
    def next_num():
        for i in range(10):
            yield i
    it = next_num()
    l = [x for x in it]
    assert l == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

"""
Item 36: Use itertools for working with iterators and generators
"""
def test_itertools():
    import itertools
    it = itertools.permutations([1, 2, 3], 2)
    l = list(it)
    assert l == [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]

    it = itertools.combinations([1, 2, 3], 2)
    l = list(it)
    assert l == [(1, 2), (1, 3), (2, 3)]

    it = itertools.combinations_with_replacement([1, 2, 3], 2)
    l = list(it)
    assert l == [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]