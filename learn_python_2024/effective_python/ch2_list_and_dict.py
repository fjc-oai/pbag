
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