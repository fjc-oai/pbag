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
2. set defaultdict and setdefault to handle missing values
3. OrderedDict, Counter are handy dict variants
4. Handy set opertions for & and |
