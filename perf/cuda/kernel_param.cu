// -----------------------------------------------------------------------------
// Summary: Major ways to pass parameters from C++ to CUDA kernels
// -----------------------------------------------------------------------------
//
// 1. **Pass by value (small POD structs)** 
//    - For small, trivially-copyable structs (no pointers, STL, or variable size),
//      you can pass the struct directly as a kernel argument.
//    - CUDA copies the struct's bytes into the kernel's parameter space (on modern
//      GPUs this is stored in constant memory, with size limits ~32KB as of CUDA 12.1).
//    - Example: `kernel<<<1,1>>>(SmallParam p, ...)`
//
// 2. **Pass by pointer (arrays or larger fixed-size data)** 
//    - For multiple elements or larger objects, allocate them in **device memory** 
//      (or pinned mapped host memory), then pass a `T*` pointer to the kernel.
//    - The kernel receives a device pointer and can index into it normally.
//    - This avoids parameter space size limits and allows dynamic sizes.
//    - Example: `kernel<<<...>>>(SmallParam* p, int n)`
//
// 3. **Pass serialized blobs for complex or variable-size structs** 
//    - For large or non-POD data, serialize the structure into a flat byte buffer 
//      (`uint8_t` array), copy it to device memory, and pass a pointer to that blob.
//    - Inside the kernel, reinterpret or parse the blob to reconstruct metadata.
//    - This is the common pattern for metadata like TensorSharding, where each 
//      entry may have variable length.
//
// 4. **Pass pointer tables (pointer-to-pointer)** 
//    - For collections of multiple blobs or descriptors, allocate each blob 
//      on device, then build a **device array of device pointers**.
//    - Pass that array’s pointer to the kernel (`const T* const*`), so the kernel 
//      can index and dereference per item.
//    - Example: `kernel<<<...>>>(const uint8_t* const* table, int idx)`
//
// These four patterns cover almost all real CUDA code:
//   - small structs → by value
//   - arrays / big fixed-size → device pointer
//   - variable-size metadata → serialized blob + device pointer
//   - many blobs → device pointer table (array-of-pointers)
//
// For performance, prefer by-value only for small PODs, and use serialization or
// pointer passing for anything larger or dynamic.
// 
// To run the code: 
//      nvcc -O2 -std=c++17 kernel_param.cu -o a.out && ./a.out
// -----------------------------------------------------------------------------
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#define CUDA_CHECK(expr)                                                              \
  do {                                                                                \
    cudaError_t _err = (expr);                                                        \
    if (_err != cudaSuccess) {                                                        \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, \
                   cudaGetErrorString(_err));                                         \
      std::exit(1);                                                                   \
    }                                                                                 \
  } while (0)

struct SmallParam {
  int a;
  int b;
};

__global__ void kernel_pass_small_param_by_value(SmallParam p, int* out) {
  if (threadIdx.x == 0) {
    out[0] = p.a + p.b;
  }
}

void run_small_param_by_value() {
  SmallParam p{1, 2};
  int* out = nullptr;
  cudaMalloc(&out, sizeof(int));
  kernel_pass_small_param_by_value<<<1, 1>>>(p, out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  int* out_host = nullptr;
  cudaMallocHost(&out_host, sizeof(int));
  cudaMemcpy(out_host, out, sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << "out: " << *out_host << std::endl;
  cudaFree(out);
  cudaFreeHost(out_host);
}

__global__ void kernel_pass_small_params_by_pointer(SmallParam* p, int* out, int n) {
  const auto tid = threadIdx.x;
  if (tid < n) {
    out[tid] = p[tid].a + p[tid].b;
  }
}

void run_small_params_by_pointer() {
  const int n = 10;
  SmallParam* p = nullptr;
  cudaMallocHost(&p, sizeof(SmallParam) * n);
  for (int i = 0; i < n; i++) {
    p[i] = SmallParam{i, i};
  }
  int* out = nullptr;
  cudaMallocHost(&out, sizeof(int) * n);
  kernel_pass_small_params_by_pointer<<<1, 32>>>(p, out, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  for (int i = 0; i < n; i++) {
    std::cout << "out[" << i << "]: " << out[i] << std::endl;
  }
  cudaFreeHost(p);
  cudaFreeHost(out);
}

#define N 4
struct LargeParam {
  int a[N];
  int b[N];

  std::vector<uint8_t> serialize() const {
    std::vector<uint8_t> data;
    data.resize(sizeof(int) * 8);
    std::cout << "sizeof(a): " << sizeof(a) << " sizeof(b): " << sizeof(b) << std::endl;
    std::memcpy(data.data(), a, sizeof(a));
    std::memcpy(data.data() + sizeof(a), b, sizeof(b));
    return data;
  }
};

__global__ void kernel_pass_large_param_by_pointer(LargeParam* p, int* out, int n) {
  const auto tid = threadIdx.x;
  if (tid < n) {
    out[tid]= p->a[tid] + p->b[tid];
  }
}

void run_large_param_by_pointer() {
  LargeParam p;
  for (int i = 0; i < N; i++) {
    p.a[i] = i;
    p.b[i] = i;
  }
  std::vector<uint8_t> data = p.serialize();
  uint8_t* d_data = nullptr;
  cudaMalloc(&d_data, data.size());
  cudaMemcpy(d_data, data.data(), data.size(), cudaMemcpyHostToDevice);
  int* out = nullptr;
  cudaMalloc(&out, sizeof(int) * N);
  kernel_pass_large_param_by_pointer<<<1, 32>>>(reinterpret_cast<LargeParam*>(d_data), out, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  int* out_host = nullptr;
  cudaMallocHost(&out_host, sizeof(int) * N);
  cudaMemcpy(out_host, out, sizeof(int) * N, cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; i++) {
    std::cout << "out[" << i << "]: " << out_host[i] << std::endl;
  }
  cudaFreeHost(out_host);

}

int main() {
  std::cout << "Running small param by value" << std::endl;
  run_small_param_by_value();
  std::cout << "Running small params by pointer" << std::endl;
  run_small_params_by_pointer();
  std::cout << "Running large param by pointer" << std::endl;
  run_large_param_by_pointer();
  return 0;
}