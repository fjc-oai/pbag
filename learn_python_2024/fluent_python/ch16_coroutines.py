import inspect

import pytest


def coro():
    x = 0
    while True:
        y = yield x
        if y is None:
            break
        x = y + 1

def test_states():
    c = coro()
    assert inspect.getgeneratorstate(c) == 'GEN_CREATED'

    next(c)
    assert inspect.getgeneratorstate(c) == 'GEN_SUSPENDED'

    c.send(1)
    assert inspect.getgeneratorstate(c) == 'GEN_SUSPENDED'

    with pytest.raises(StopIteration):
        c.send(None)
    assert inspect.getgeneratorstate(c) == 'GEN_CLOSED'

def test_interaction():
    c = coro()
    x = next(c)
    assert x == 0

    x = c.send(1)
    assert x == 2

    x = c.send(10)
    assert x == 11

    with pytest.raises(StopIteration):
        c.send(None)
        
def averager():
    total = 0.0
    count = 0
    while True:
        item = yield total / count if count > 0 else 0
        total += item
        count += 1

def test_averager():
    avg = averager()
    next(avg)
    assert avg.send(10) == 10
    assert avg.send(20) == 15
    assert avg.send(30) == 20


def test_stop():
    avg = averager()
    next(avg)
    assert avg.send(10) == 10
    avg.close()
    with pytest.raises(StopIteration):
        avg.send(20)