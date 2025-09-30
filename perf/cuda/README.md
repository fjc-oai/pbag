# Passing Parameters from C++ to CUDA Kernels

This example demonstrates the major ways to pass data from C++ host code to CUDA kernels.

## 1. Pass by Value (Small POD Structs)

For small, trivially copyable structs (no pointers, STL containers, or variable size fields),
you can pass the struct **by value** as a kernel argument.  

CUDA copies the struct's bytes into the kernel's **parameter space** (typically constant memory).  
This is efficient for small objects and limited by the kernel argument size limit  
(~32 KB as of CUDA 12.1).

---

## 2. Pass by Pointer (Arrays or Larger Fixed-Size Data)

For larger objects or arrays, allocate them in **device memory** (or pinned mapped host memory)  
and pass a device pointer to the kernel. The kernel receives the pointer and can index into it.

This avoids parameter size limits and works well for arrays and structs of moderate size.

---

## 3. Pass Serialized Blobs (For Complex / Variable-Size Structs)

For large or non-POD structures, serialize the object into a **flat byte buffer** (`uint8_t[]`),  
copy that to device memory, and pass a pointer to the blob.

Inside the kernel, parse or reinterpret the blob to reconstruct the structure.  
This is useful for variable-length metadata (e.g. TensorSharding), where each entry may have a different size.

---

## 4. Pass Pointer Tables (Pointer-to-Pointer)

For collections of multiple blobs or descriptors, allocate each blob on device,  
then build a **device array of device pointers** and pass that array to the kernel.

This lets the kernel index and dereference individual descriptors efficiently.

---

## Summary

| Pattern                       | Use Case                                 | Notes |
|-------------------------------|-------------------------------------------|-------|
| **By value**                  | Small POD structs                        | Copied into kernel param space (fast) |
| **Pointer**                   | Arrays or larger fixed-size data         | Allocate on device, pass pointer |
| **Serialized blob + pointer** | Large or variable-size structs           | Serialize, copy to device, parse on kernel |
| **Pointer table**             | Collections of blobs/descriptors         | Pass device array of device pointers |

**Rule of thumb:**  
- Use **by value** for very small, fixed-size structs.  
- Use **pointers** or **serialized blobs** for anything larger or dynamic.  
- Use **pointer tables** for collections of heterogeneous descriptors.

---

### Build & Run (example)

```bash
nvcc -O2 -std=c++17 kernel_param.cu -o a.out && ./a.out
```
