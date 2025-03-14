# To build: 
#   python setup.py build_ext --inplace
import pybind11
from setuptools import Extension, setup

# Define the cprofiler extension module (your existing C++ code)
cprofiler_module = Extension(
    "cprofiler",
    sources=["cprofiler.cpp"],
    extra_compile_args=["-std=c++11"],
    language="c++",
)

setup(
    name="my_profiling_tools",
    version="0.1",
    description="A collection of C++ extensions for profiling and CPU burning.",
    ext_modules=[
        cprofiler_module,
    ],
)
