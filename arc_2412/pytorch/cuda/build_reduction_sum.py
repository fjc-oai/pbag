# python build_reduction_sum.py build_ext --inplace

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="reduction_sum_ext",
    ext_modules=[
        CUDAExtension(
            name="reduction_sum_ext",
            sources=["reduction_sum_kernel.cu"],
            include_dirs=["path/to/pybind11/include"],  # Update this path
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--expt-relaxed-constexpr"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
