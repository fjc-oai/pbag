"""
Item 38: Accept Functions Instead of Classes for Simple Interfaces
"""
from typing import Any


def test_accept_functions():
    def get_age(x):
        return x['age']
    
    people = [{'name': 'John', 'age': 30}, {'name': 'Jane', 'age': 15}, {'name': 'Doe', 'age': 40}, {'name': 'Smith', 'age': 11}] 
    people.sort(key=get_age)
    assert people == [{'name': 'Smith', 'age': 11}, {'name': 'Jane', 'age': 15}, {'name': 'John', 'age': 30}, {'name': 'Doe', 'age': 40}]

    # Stateful closure
    n_child = 0
    def get_age_while_counting(x):
        nonlocal n_child
        age = x['age']
        if age < 18:
            n_child += 1
        return age
    people = [{'name': 'John', 'age': 30}, {'name': 'Jane', 'age': 15}, {'name': 'Doe', 'age': 40}, {'name': 'Smith', 'age': 11}]
    people.sort(key=get_age_while_counting)
    assert people == [{'name': 'Smith', 'age': 11}, {'name': 'Jane', 'age': 15}, {'name': 'John', 'age': 30}, {'name': 'Doe', 'age': 40}]
    assert n_child == 2

    # Class
    class ChildCounter():
        def __init__(self):
            self.n_child = 0

        def __call__(self, x):
            age = x['age']
            if age < 18:
                self.n_child += 1
            return age
    people = [{'name': 'John', 'age': 30}, {'name': 'Jane', 'age': 15}, {'name': 'Doe', 'age': 40}, {'name': 'Smith', 'age': 11}]
    cc = ChildCounter()
    people.sort(key=cc)
    assert people == [{'name': 'Smith', 'age': 11}, {'name': 'Jane', 'age': 15}, {'name': 'John', 'age': 30}, {'name': 'Doe', 'age': 40}]
    assert cc.n_child == 2

"""
Item 40: Initialize Parent Classes with super
"""
def test_super():
    class Base:
        def __init__(self):
            self.n = 3
    
    class TimesTwo(Base):
        def __init__(self):
            super().__init__()
            self.n *= 2

    class PlusFive(Base):
        def __init__(self):
            super().__init__()
            self.n += 5
    
    class GoodWay(TimesTwo, PlusFive):
        def __init__(self):
            super().__init__()

    gw = GoodWay()
    assert gw.n == 16
    assert list(GoodWay.mro()) == [GoodWay, TimesTwo, PlusFive, Base, object]


"""
Item 41: Consider Composing Functionality with Mix-in Classes
"""
def test_mixin():
    import torch
    class CpuTensorMixin:
        def count(self):
            return len([param for param in self.parameters() if param.is_cpu])

    class Attn(torch.nn.Module, CpuTensorMixin):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Linear(10, 10, bias=False)
            self.proj = torch.nn.Linear(10, 10, bias=False)
    
    attn = Attn()
    assert attn.count() == 2

"""
Item 43: Inherit from collections.abc for Custom Container Types
"""
def test_collections_abc():
    from collections.abc import Sequence
    class BusSchedule(Sequence):
        def __init__(self, arrival_times):
            self._arrival_times = list(arrival_times)
        
        def __getitem__(self, index):
            return self._arrival_times[index]
        
        def __len__(self):
            return len(self._arrival_times)
        
    schedule = BusSchedule([1, 2, 3, 4, 5])
    assert schedule[0] == 1
    assert len(schedule) == 5