# Not really working, yet, loool
import numpy as np
import reduction_sum_ext

# Sample data
N = 1024
input_array = np.ones(N, dtype=np.float32)

# Allocate output array
blocks_per_grid = 4  # Adjust based on your needs
output_array = np.empty(blocks_per_grid, dtype=np.float32)

# Call the CUDA kernel
reduction_sum_ext.reduction_sum(input_array, output_array, N)
print("Output array:", output_array)
# Compute the final sum
sum_result = np.sum(output_array)

# Print the result
print("Sum:", sum_result)
