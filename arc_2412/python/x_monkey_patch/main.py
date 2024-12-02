"""
(From ChatGPT) How does monkey patching work under the hood, e.g. manipulating
the target module in one file, the effect can be affected in all other files.


Python's import system

1. When a module is imported for the first time, Python executes the module's
   code and creates a single instance of it.

2. This instance is stored in the sys.modules dictionary, which maps module
   names to module objects.

3. Subsequent imports of the same module reference the already-loaded module
   object from sys.modules rather than reloading the module. This ensures that
   there is only one instance of each module, preserving changes made to it.

Object References and Mutability

1. In Python, variables hold references to objects rather than the objects
   themselves.

2. When you modify an attribute of a module (like a function or a class), you're
   modifying the object that the attribute references.

3. Any other code that has a reference to that object will see the changes
   because the reference itself remains the same. 


"""
import patcher
from utils import call_foo


def main():
    print(call_foo())

if __name__ == '__main__':
    main()