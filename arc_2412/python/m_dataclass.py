from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Literal


@dataclass
class Prompt:
    type: Literal["text", "token"]
    data: str | list[int] = field(default_factory=list)
    encoding: str = "utf-8"


def test_replace():
    p = Prompt("text", "data")
    q = replace(p, encoding="another_encoding")
    assert p.type == q.type
    assert p.data == q.data
    q.data = "another_data"
    assert p.data != q.data

    p = Prompt("token", [1, 2, 3])
    q = replace(p, encoding="another_encoding")
    assert p.type == q.type
    assert p.data == q.data
    q.data.append(4)
    assert p.data == q.data

    print("test_replace passed")


def test_deepcopy():
    p = Prompt("token", [1, 2, 3])
    q = deepcopy(p)
    assert p.type == q.type
    assert p.data == q.data
    assert p.encoding == q.encoding
    q.data.append(4)
    assert p.data != q.data

    print("test_deepcopy passed")


def main():
    test_replace()
    test_deepcopy()


if __name__ == "__main__":
    main()
