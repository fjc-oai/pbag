Fluent Python, by Luciano Ramalho
https://www.amazon.com/Fluent-Python-Concise-Effective-Programming/dp/1491946008

###################################################
# Chapter 1: python data model
###################################################
1. ** Dunder method (magic method)
    - Python built-in method, standard libs are built upon dunder method.
    - Define dunder method for your class. Get all functionalities for free.


###################################################
# Chapter 2: an array of sequences
###################################################
1. list operations: +, *
2. array stores packed bytes, which is more effective than list
3. Use memoryview() to inspect memory buffer with cast()


###################################################
# Chapter 3: dictionaries and sets
###################################################
1. ** hashable
    - Hashable-ness can be used as dict/set key
    - Hashable requires to define __hash__() and __eq__(). Equal objects much have the same hash.
    - Custom classes are by default hashable. __hash__() returns id(), and __eq__() returns False.
2. Use defaultdict and setdefault to handle missing values
3. OrderedDict, Counter are handy dict variants
4. Handy set opertions for & and |


###################################################
# Chapter 4. str vs bytes
###################################################
1. Unicode: separation of code point and byte representation
    - Encode: code point to bytes
    - Decode: bytes to code point
    - Str in RAM representation is Python implementation details. Usually some memory efficient format
2. Bytes display
    - Printable ASCII bytes: displayed as is
    - Special chars, e.g. tab, newline: using escape sequences, e.g. \t, \n, etc
    - Other bytes: hexadecimal escape sequence, e.g. \x00
3. Unicode sandwidth
    - Decode bytes on input, process text only, encode text on output
    - open() handles encoding/decoding automatically 
        - w/r: open files in text mode, with default uft-8 encoding/decoding. str in and str out. 
        - raises exception if, say, write bytes
        - wb/rb: open files in byte mode. expect bytes input/output

###################################################
# Chapter 5. First Class Functions
###################################################
0. First class object:
    - created at runtime, passed as an argument to a function, returned as a result from a function
1. dir() vs __dict__(): 
    - dir() returns all attributes?
    - __dict__() returns user attributes assigned to it?
2. Inspection
    - __names__, __doc__, __annotation__, __code__, etc
    - inspect module
3. Handy higher-order functions
    - partial: freeze arguments

###################################################
# Chapter 7. Decorators and Closures
###################################################
1. Variable scope
    - Python compiles the body of the function before execution
    - If a variable is assigned to a value, Python treats it as a new variable
    - If a variable is accessed only, it look up outer scopes as a reference
    - Use dis() to disassemble a function and inspect bytecode
2. Closure: a function with an extended scope 
    - Access non-global variables that defined out of its body
    - Called free variables
3. nonlocal: declare reference to a free variable
4. Decorator & Parameterized Decorator
    - Use functools.wraps to copy relavant attributes
    - Parameterized decorator needs to be a decorator factory
    - Another handy functools lib, lru_cache

###################################################
# Chapter 8. Object Memory Management
###################################################    
1. Identity, equality, and alias
    - is vs ==
2. Copy vs deepcopy
    - Shallow copy by default
    - Shallow copy creates alias for each attribute of the object
        - For list, it creates alias for each element
        - For dict, it creates alias for each kv
    - deepcopy recursively copies everything
3. Function parameters as references
    - Mutable types as parameter defaults is error prone, e.g. def foo(l=[])
        - All class/function instances share the same default param value
    - Use copy instead of assign to store argument as member variable, e.g.
        def foo(self, l):
            self._l = list(l)
4. Gargabe Collection
    - del deletes names, not objects
    - Objects are freed either, 1) refcount reaches zero, immediately destroyed, 2) reference cycle detection when gc.collect()
    - Python console automatically bind _ variable to the result of expression that are not None
    
###################################################
# Chapter 9. A Pythonic Object (more dunder methods)
###################################################    
1. Classmethod vs Staticmethod: 
    - Classmethod: commonly used as alternative constructors
    - Staticmethod: no good reason of existance
2. More dunder methods: __format__(), __hash__()
    - __slots__: 1) efficient memory format, 2) forbid extra attributes definitions

###################################################
# Chapter 10 & 11. More and more dunner methods
###################################################    
1. Duck-typing: don't check type. Check behaviors.
2. Monkey patching: changing a class or module at runtime, without touching the source code.

###################################################
# Chapter 14. Iterables, Iterators, and Generators
###################################################    
1. For-in-loop under the hook  
    - calls iter(x) over an object to get the iterator
    - iter() checks if __iter__() is implemented, or fallback to __getitem__(), or TypeError
    - repeatedly calls next(it), until StopIteration exception
2. Iterable vs Iterator
    - Iterable interface implements __iter__() method which returns a iterator, (or implements __getitem__())
    - Iterator inteface implements __next__() to return the next available item (or raise StopIteration), 
        and __iter__() to return itself, which allows iterators to be used where an iterable is expected
    Iterators are iterable. Iterables may not be iterators.
3. Generator vs Iterator
    - Functional wise, every generator is a iterator, which implements iterator interface (__next__ and __iter__).
    - Conceptual wise, iterator retrieves items from an existing inventory, whereas generator creates new things.
    - In many cases, people don't strictly distinsh iterator and generator.
4. itertools.count(0, 1)

###################################################
# Chapter 15. Contgext Manasger
################################################### 
1. Implement __enter__() and __exit__() for context manager interface.
2. Use @contextlib.contextmanager decorator with yield.
3. Remember to include try final, otherwise exception raised in the body of with block, without restoring the state.

###################################################
# Chapter 16. Coroutines
################################################### 
1. Coroutines for Cooperative Multitask
2. Coroutine has 4 state: created, running, suspended, closed
3. Push and pull values from coroutine: x = yield y
4. Major interface: next(), send(), close(), throw()
5. yield from: a syntax to allow the client to directly drive subgenerator directly, effectively bypass delegating generators