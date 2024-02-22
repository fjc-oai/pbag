def test_defaultdict():
    from collections import defaultdict
    d = defaultdict(lambda: defaultdict(list))
    d['steve']['friends'].append('bob')
    assert len(d['steve']['friends']) == 1

def test_dict():
    d = {}
    r = d.get('steve', [])
    assert isinstance(r, list)
    r.append(7)
    assert len(r) == 1
    assert len(d) == 0

