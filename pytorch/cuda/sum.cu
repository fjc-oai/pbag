#include <stdio.h>

__global__ void reduction_sum(float *input, float *output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to output
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    int n = 1024;
    float *input, *output;
    float *d_input, *d_output;

    // Allocate host memory
    input = (float*)malloc(n * sizeof(float));
    output = (float*)malloc((n / 256) * sizeof(float)); // Assuming 256 threads per block

    // Initialize input array
    for (int i = 0; i < n; ++i) {
        input[i] = 1.0f; // All elements are 1 for simplicity
    }

    // Allocate device memory
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, (n / 256) * sizeof(float));

    // Copy input array to device
    cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block sizes
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    reduction_sum<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>(d_input, d_output, n);

    // Copy the result back to the host
    cudaMemcpy(output, d_output, (n / 256) * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute final sum on the host
    float sum = 0.0f;
    for (int i = 0; i < blocks_per_grid; ++i) {
        sum += output[i];
    }

    // Print the result
    printf("Sum: %f\n", sum);

    // Free memory
    free(input);
    free(output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
