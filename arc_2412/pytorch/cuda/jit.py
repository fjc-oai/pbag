# NOT WORKING.... :(
import torch
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel using a string
cuda_source = """
extern "C" __global__
void vector_add(const float* a, const float* b, float* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}
"""

# Compile and load the CUDA kernel
vector_add_module = load_inline(
    name="vector_add_module", cpp_sources="", cuda_sources=cuda_source, functions=["vector_add"]
)

# Get the compiled function
vector_add = vector_add_module.vector_add

# Sample data
N = 1024
a = torch.rand(N, device="cuda", dtype=torch.float32)
b = torch.rand(N, device="cuda", dtype=torch.float32)
c = torch.empty_like(a)

# Define grid and block sizes
threads_per_block = 256
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

# Launch the kernel
vector_add(a, b, c, N, block=(threads_per_block, 1, 1), grid=(blocks_per_grid, 1, 1))

# Verify the result
expected = a + b
assert torch.allclose(c, expected)
print("Vector addition successful!")
