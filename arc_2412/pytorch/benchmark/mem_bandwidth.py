import torch
import time


def mb(n):
    return f"{n / 1024 ** 2:.2f} MB"

def gb(n):
    return f"{n / 1024 ** 3:.2f} GB"

# HBM size: 39.39 GB
print(f"HBM size: {gb(torch.cuda.get_device_properties(0).total_memory)}")

# https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
N_SM = 108
N_REG_PER_SM = 256 * 1024
REG_SIZE = 4 # 32 bits
# Register size: 108.00 MB
print(f"Register size: {mb(N_SM * N_REG_PER_SM * REG_SIZE)}") # 

device = torch.device('cuda:0')
tensor_size = 1024 * 1024 * 64  # This creates a tensor of size 256MB
dtype = torch.float32  # Specify the data type

a = torch.rand(tensor_size, dtype=dtype, device=device)
b = torch.rand(tensor_size, dtype=dtype, device=device)

# Warm-up
for _ in range(10):
    c = a + b

torch.cuda.synchronize()  # Ensure CUDA operations are completed

# Start measuring time
start_time = time.time()

# Perform operations
N_ITER = 10000
for _ in range(N_ITER):
    c = a + b

torch.cuda.synchronize()  # Ensure CUDA operations are completed

# Measure the elapsed time
elapsed_time = time.time() - start_time

# Compute total data processed: 3 tensors per operation * size * number of operations
total_data = 3 * tensor_size * torch.tensor(1).element_size() * N_ITER

# Convert to GB
total_data_gb = total_data / (1024 ** 3)

# Bandwidth in GB/s
bandwidth = total_data_gb / elapsed_time

# Elapsed time for 100 operations: 5.97 seconds
# Total data processed: 15000.00 GB
# Memory Bandwidth: 2514.35 GB/s
print(f"Elapsed time for {N_ITER} operations: {elapsed_time:.2f} seconds")
print(f"Total data processed: {total_data_gb:.2f} GB")
print(f"Memory Bandwidth: {bandwidth:.2f} GB/s")
