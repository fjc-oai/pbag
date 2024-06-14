import array
import sys


def test_list():
    l = [1, 2, 3]
    assert l + [4, 5] == [1, 2, 3, 4, 5]
    assert l * 2 == [1, 2, 3, 1, 2, 3]


def test_array():
    arr = array.array("i", [1, 2, 3])
    assert arr[0] == 1


def test_memoryview():
    a = array.array("B", [0, 0, 0, 0]) # signed char, 1 byte
    mv = memoryview(a)
    mv2 = mv.cast("H") # signed short, 2 bytes
    mv2[0] = 258 # big endian, 1 * 256 + 2
    assert a == array.array("B", [2, 1, 0, 0])
        
