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