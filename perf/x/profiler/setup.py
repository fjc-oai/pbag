import pybind11
from setuptools import Extension, setup

# Define the cprofiler extension module (your existing C++ code)
cprofiler_module = Extension(
    "cprofiler", sources=["cprofiler.cpp"], extra_compile_args=["-std=c++11"], language="c++"
)

# Define the burn_module extension module (with pybind11 bindings)
burn_module = Extension(
    "burn_module",
    sources=["burn.cpp"],
    include_dirs=[pybind11.get_include()],
    extra_compile_args=["-std=c++11"],
    language="c++",
)

# Define the cpython_thread_profiler extension module.
cpython_thread_profiler_module = Extension(
    "cpython_thread_profiler",
    sources=["cpython_thread_profiler.cpp"],
    extra_compile_args=["-std=c++11"],
    language="c++",
)

setup(
    name="my_profiling_tools",
    version="0.1",
    description="A collection of C++ extensions for profiling and CPU burning.",
    ext_modules=[cprofiler_module, burn_module, cpython_thread_profiler_module],
)
