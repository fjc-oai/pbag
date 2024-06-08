"""
Chapter 2: Lists and Dictionaries

1. catch-all unpacking
    a, *b, c = [1, 2, 3, 4, 5]

2. sort using key parameter
    sorted(iterable, key=lambda x: x[1])

3. update nested dict
    d[key] = d.get(key, 0) + 1
    from collections import defaultdict
    d = defaultdict(lambda: defaultdict(list))

"""


"""
Item 13: Prefer Catch-All Unpacking Over Slicing
"""
import random


def test_star_expression():
    l = [random.random() for _ in range(10)]
    first, *middle, last = l
    assert first == l[0]
    assert last == l[-1]
    assert middle == l[1:-1]

"""
Item 14: Sort by Complex Criteria Using the key Parameter
"""
def test_sort_by_key():
    prices = {"apple": 1.99, "banana": 0.99, "cherry": 2.99}
    sorted_prices = sorted(prices.items(), key=lambda x: x[1])
    assert sorted_prices == [("banana", 0.99), ("apple", 1.99), ("cherry", 2.99)]


""" 
Item 16: Prefer get Over in and KeyError to Handle Missing Dictionary Keys
"""

def test_get_over_in():
    # dict[key, int]
    d = {}
    keys = ["apple", "banana", "cherry", "apple", "banana"]
    for key in keys:
        d[key] = d.get(key, 0) + 1
    assert d == {"apple": 2, "banana": 2, "cherry": 1}

    # dict[key, list]
    d = {}
    pairs = [("apple", 1), ("banana", 1), ("cherry", 1), ("apple", 2), ("banana", 2)]
    for key, value in pairs:
        if (val := d.get(key)) == None:
            d[key] = val = []
        val.append(value)
    assert d == {"apple": [1, 2], "banana": [1, 2], "cherry": [1]}

    # dict[dict[str, list[int]]]
    d = {}
    d2 = d.setdefault('k1', {})
    l = d2.setdefault('k2', [])
    l.append(1)

    d2 = d.setdefault('k1', {})
    l = d2.setdefault('k3', [])
    l.append(2)

    d2 = d.setdefault('k1', {})
    l = d2.setdefault('k2', [])
    l.append(3)
    assert d == {'k1': {'k2': [1, 3], 'k3': [2]}}

"""
Item 17: Prefer defaultdict Over setdefault to Handle Missing Items in Internal State
"""
def test_defaultdict_over_setdefault():
    d = {}
    d.setdefault('k1', {}).setdefault('k2', []).append(1)
    d.setdefault('k1', {}).setdefault('k3', []).append(2)
    d.setdefault('k1', {}).setdefault('k2', []).append(3)
    assert d == {'k1': {'k2': [1, 3], 'k3': [2]}}

    from collections import defaultdict
    d = defaultdict(lambda: defaultdict(list))
    d['k1']['k2'].append(1)
    d['k1']['k3'].append(2)
    d['k1']['k2'].append(3)
    assert d == {'k1': {'k2': [1, 3], 'k3': [2]}}