#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>

namespace py = pybind11;
extern "C" __global__
void reduction_sum_kernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Wrapper function for the CUDA kernel
void reduction_sum(const py::array_t<float>& input_array, py::array_t<float>& output_array, int n) {
    auto input_buf = input_array.request();
    auto output_buf = output_array.request();

    const float* input = static_cast<float*>(input_buf.ptr);
    float* output = static_cast<float*>(output_buf.ptr);

    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    reduction_sum_kernel<<<blocks_per_grid, threads_per_block>>>(input, output, n);
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(reduction_sum_ext, m) {
    m.def("reduction_sum", &reduction_sum, "Reduction sum kernel");
}