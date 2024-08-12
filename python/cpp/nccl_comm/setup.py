"""
Setup tools script for help building the package with C++ bindings.

It seems pyproject.toml is the latest way to build packages but setup.py still
provides handy features for building packages with C++ bindings.

python setup.py build
python setup.py install
python setup.py clean --all
"""

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup
from setuptools.command.build_ext import build_ext

ext_modules = [
    Pybind11Extension(
        "nccl_comm.ops",
        sources=["src/ops.cpp"],
        language="c++",
        extra_compile_args=["-std=c++11"],
    ),
]

setup(
    name="nccl_comm",
    version="0.1.0",
    author="fmars",
    author_email="fmarsf@gmail.com",
    description="A package with C++ bindings",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
