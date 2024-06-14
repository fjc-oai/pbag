from dataclasses import dataclass
import random
from typing import Sequence


def test_dunder_method():
    @dataclass
    class Student:
        name: str
        Age: int

    class Class:
        def __init__(self, students: Sequence[Student]):
            self.students = list(students)

        def __len__(self):
            return len(self.students)

        def __getitem__(self, i: int):
            return self.students[i]

    students = [Student("Alice", 10), Student("Bob", 12), Student("Charlie", 11)]
    c = Class(students)

    assert len(c) == 3
    assert c[0] == Student("Alice", 10)
    assert c[-1] == Student("Charlie", 11)

    c2 = sorted(c, key=lambda s: s.Age)
    assert c2[:] == [Student("Alice", 10), Student("Charlie", 11), Student("Bob", 12)]

    student = random.choice(c)
    assert student in c
