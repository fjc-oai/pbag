// nvcc vector_add.cu -o vector_add
// ./vector_add

#include <stdio.h>

__global__ void vector_add(float *a, float *b, float *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int n = 1024;
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;

    // Allocate host memory
    a = (float*)malloc(n * sizeof(float));
    b = (float*)malloc(n * sizeof(float));
    c = (float*)malloc(n * sizeof(float));

    // Initialize host arrays
    for (int i = 0; i < n; ++i) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Allocate device memory
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    // Copy host arrays to device
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block sizes
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    vector_add<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, n);

    // Copy the result back to the host
    cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < 10; ++i) {
        printf("c[%d] = %f\n", i, c[i]);
    }

    // Free memory
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
