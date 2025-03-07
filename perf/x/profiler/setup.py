from setuptools import Extension, setup

# Define the extension module
module = Extension(
    'cprofiler',            # Name of the module
    sources=['cprofiler.cpp'], # List of source files
    extra_compile_args=['-std=c++11'],
    language='c++'
)

# Call setup to compile and package the module
setup(
    name='cprofiler',
    version='0.1',
    description='A simple C++ level profiler using the Python C API.',
    ext_modules=[module],
)