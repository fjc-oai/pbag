"""
Know the Differences Between bytes and str 

- bytes contains sequences of 8-bit values, and str contains sequences of
  Unicode code points.

- str instances can be encoded to bytes, and bytes instances can be decoded to
  str.

- Python internally uses a Flexible String Representation to represent str,
  instead of UTF-8 or UTF-16, etc.

- When read and write files, use the 'b' flag to open files in binary
"""

def test_bytes_and_str():
    x = '你好'
    y = x.encode('utf-8')
    z = y.decode('utf-8')
    assert type(x) == str
    assert type(y) == bytes
    assert type(z) == str
    assert x == z

    path = '/tmp/binary_file'
    with open(path, 'wb') as f:
        f.write(y)
    with open(path, 'rb') as f:
        data = f.read()
    assert type(data) == bytes
    assert data == y


"""
Prefer f-string formatting to the % operator
"""

def test_f_string_formatting():
    print(f"|{'hello':>10}|{'world':<10}|")

"""
Prefer multiple assignment unpacking over indexing

"""

def test_multiple_assignment_unpacking():
    names = ['Alice', 'Bob', 'Charlie']
    a, b, c = names
    assert a == 'Alice'

    prices = {'apple': 1.99, 'banana': 0.99, 'cherry': 2.99}
    for fruit, price in prices.items():
        print(f"{fruit}: {price}")

"""
Prefer enumerate over range
"""

def test_enumerate_over_range():
    names = ['Alice', 'Bob', 'Charlie']
    for i, name in enumerate(names):
        print(f"{i}: {name}")



"""
Use zip to process iterators in parallel

- zip stops when the shortest input iterator is exhausted
- use itertools.zip_longest to continue processing until the longest iterator
"""

def test_zip():
    names = ['Alice', 'Bob', 'Charlie']
    ages = [20, 30, 40]
    for name, age in zip(names, ages):
        print(f"{name}: {age}")

    names += ['David']
    import itertools
    for name, age in itertools.zip_longest(names, ages):
        print(f"{name}: {age}")


"""
Prevent Repetition with Assignment Expressions (walrus operator)

- Wrap an assignment in parentheses and the value of the assignment will be
  equal to the value assigned.
"""

def test_assignment_expressions():
    def get_a_num():
        return 7
    if (num := get_a_num()) > 5:
        print(f"num is {num}")
    print(f"again, num is {num}")
    