import itertools
import re
from types import FunctionType, GeneratorType

import pytest

RE_WORD = re.compile('\w+')

# __getitem__() for iterator
class Sentence:
    def __init__(self, text):
        self.text = text
        self.words = RE_WORD.findall(text)

    def __getitem__(self, index):
        return self.words[index]

    def __len__(self):
        return len(self.words)
    

# __iter__() for iterator
class SentenceIterator:
    def __init__(self, words):
        self.words = words
        self.index = 0
    
    def __next__(self):
        if self.index < len(self.words):
            word = self.words[self.index]
            self.index += 1
            return word
        raise StopIteration()

    def __iter__(self):
        return self
    
class Setence2:
    def __init__(self, text):
        self.text = text
    
    def __iter__(self):
        return SentenceIterator(RE_WORD.findall(self.text))
    
# generator function
class Sentence3:
    def __init__(self, text):
        self.text = text
        self.words = RE_WORD.findall(text)
    
    def __iter__(self):
        for word in self.words:
            yield word


# lazy generator
class Sentence4:
    def __init__(self, text):
        self.text = text
    
    def __iter__(self):
        for match in RE_WORD.finditer(self.text):
            yield match.group()

# generator expression
class Sentence5:
    def __init__(self, text):
        self.text = text
    
    def __iter__(self):
        return (match.group() for match in RE_WORD.finditer(self.text))
    
def run_test(cls):
    s = cls("python is the best language in the world because it")
    assert list(s) == ['python', 'is', 'the', 'best', 'language', 'in', 'the', 'world', 'because', 'it']
    itr = iter(s)
    assert next(itr) == 'python'
    assert next(itr) == 'is'
    itr2 = iter(s)
    assert next(itr2) == 'python'
    assert list(itr2) == ['is', 'the', 'best', 'language', 'in', 'the', 'world', 'because', 'it']


def test_all():
    for cls in [Sentence, Setence2, Sentence3, Sentence4, Sentence5]:
        run_test(cls)
        print(f"{cls.__name__} passed")
    print("All tests passed")


def test_generator():
    def gen123():
        yield 1
        yield 2
        yield 3
    g = gen123()
    assert type(gen123) == FunctionType
    assert type(g) == GeneratorType
    x = next(g)
    assert x == 1
    x = next(g)
    assert x == 2
    x = next(g)
    assert x == 3
    with pytest.raises(StopIteration):
        next(g)

    gg = iter(g)
    assert gg is g

def test_itertools_count():
    l = []
    for i in itertools.count(0, 1):
        l.append(i)
        if i >= 10:
            break
    assert l == list(range(11))

        