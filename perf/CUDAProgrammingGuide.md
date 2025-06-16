# CUDA Programming Guide

# Chapter 3. Introduction

Programming model:

- Threading model—a hierarchy of thread groups
- Memory model—shared memories
- Communication model—barrier synchronization

# Chapter 5. Programming Model

Thread Hierarchy

- Thread blocks execute independently.
- Threads within a block cooperate by sharing data through shared memory and synchronizing execution with `__syncthreads()`.
    - Both shared memory and `__syncthreads()` are lightweight.
- Each thread block can be scheduled on any available SM. A compiled CUDA program can run on any number of SMs—only the runtime needs to know the physical SM count.

Memory Hierarchy

- Each thread has its own private local memory.
- Threads within a block share a common shared memory.
- All threads have access to global memory.

# Chapter 6. Programing Interface

- The runtime offers C/C++ functions to manage memory, launch kernels, and more.
- The Driver API provides finer control over CUDA contexts and modules.
- CUDA kernel programming uses C++ syntax, which is compiled into PTX—an architecture-independent IR—and then lowered to machine code.
- Linear memory is typically allocated with `cudaMalloc()` and freed with `cudaFree()`. Data transfers between host and device memory are usually performed using `cudaMemcpy()`.
- If a CUDA kernel repeatedly accesses a data region in global memory, these accesses are called persisting. If the data is only accessed once, these accesses are considered streaming.
- A portion of the L2 cache can be reserved for persisting global memory accesses.
- Shared memory is allocated using the __sharded__ memory space specifier.
- `cudaHostAlloc()` and `cudaFreeHost()` allocate and free page-locked host memory.
- On systems with a front-side bus, bandwidth between host and device memory is higher if host memory is page-locked, and even greater if it’s also allocated as write-combining, as described in Write-Combining Memory.
- A block of page-locked host memory can also be mapped into the device’s address space.
- At runtime, the GPU scheduler uses stream priorities to determine task execution order, though these priorities are hints rather than guarantees.
- Peer-to-peer memory access is supported between two devices if `cudaDeviceCanAccessPeer()` returns true for both.
- A unified address space is used for both devices (see Unified Virtual Address Space), so the same pointer addresses memory from either device.

# Chapter 7. Hardware Implementation

The NVIDIA GPU architecture is built around a scalable array of multithreaded Streaming Multiprocessors (SMs). 

- When a CUDA program on the host CPU launches a kernel grid, **the grid’s blocks are enumerated and assigned to multiprocessors with available execution capacity**.
- Threads within a block execute concurrently on a single multiprocessor, and multiple thread blocks can also run concurrently on one multiprocessor.
- As thread blocks complete, new blocks are dispatched to the vacated multiprocessors.multiprocessors

## SIMT Architecture

**The multiprocessor creates, manages, schedules, and executes threads in groups of 32 parallel threads called warps**. While all threads in a warp start together at the same program address, each maintains its own instruction address counter and register state—allowing them to branch and execute independently.

When a multiprocessor receives one or more thread blocks to execute, it partitions them into warps. Each warp is then scheduled for execution by a warp scheduler. The partitioning is always consistent: each warp consists of threads with consecutive, increasing thread IDs, with the first warp containing thread 0.

**A warp executes a single instruction at a time**; maximum efficiency is achieved when all 32 threads in a warp follow the same execution path. If threads diverge due to a data-dependent branch, the warp executes each path in turn, disabling threads not on the current path. Branch divergence occurs only within a warp—different warps always execute independently, regardless of whether their code paths overlap or not.

The SIMT architecture resembles SIMD (Single Instruction, Multiple Data) vector organizations, where a single instruction controls multiple processing elements. However, while SIMD exposes its width to software, SIMT instructions define the behavior of a single thread. Unlike SIMD vector machines, SIMT lets programmers write thread-level parallel code for independent scalar threads, as well as data-parallel code for groups of coordinated threads.

## Hardware Multithreading

The execution context—program counters, registers, and so on—**for each warp processed by a multiprocessor is kept on-chip for the warp’s entire lifetime. As a result, switching between execution contexts is cost-free**. At each instruction issue, the warp scheduler simply selects a warp with threads ready for their next instruction (the active threads of the warp) and issues that instruction.

Each multiprocessor contains a set of 32-bit registers partitioned among the warps, as well as a parallel data cache or shared memory, which is divided among the thread blocks.

**The number of blocks and warps** that can simultaneously reside and execute on a multiprocessor for a given kernel depends on the registers and shared memory required by the kernel, as well as the resources available on the multiprocessor.

# Chapter 8. Performance Guidelines

- At the application level, use asynchronous concurrent execution.
- Shared memory acts as a user-managed cache—the application explicitly allocates and accesses it. As shown in the CUDA Runtime, a common programming pattern is to stage data from device memory into shared memory. Typically, each thread in a block will:
▶ Load data from device memory into shared memory,
▶ Synchronize with all other threads in the block so each can safely read shared memory locations populated by different threads,
▶ Process the data in shared memory,

Shared memory

- The same on-chip memory serves both L1 and shared memory, and the allocation between the two is configurable for each kernel call. Because it is on-chip, shared memory has much higher bandwidth and much lower latency than local or global memory.
- To achieve high bandwidth, shared memory is split into equally sized modules called banks, which can be accessed at the same time. Any memory read or write involving *n* addresses that each fall into a different bank can be serviced simultaneously, resulting in total bandwidth that is *n* times that of a single module.

Flow control

- Any flow control instruction—such as if, switch, do, for, or while—can greatly affect instruction throughput by causing threads within the same warp to diverge (i.e., to follow different execution paths).
- When this occurs, the divergent paths must be serialized, increasing the total number of instructions the warp must execute.To obtain best performance in cases where the control flow depends on the thread ID, the controlling condition should be written so as to minimize the number of divergent warps.
- This is possible because the distribution of the warps across the block is deterministic as mentioned in SIMT Architecture. A trivial example is when the controlling condition only depends on (threadIdx ∕ warpSize) where warpSize is the warp size. In this case, no warp diverges since the controlling condition is perfectly aligned with the warps.

# Chapter 10. C++ Language Extensions

- __global__ executed on the device, callable from host
- __device__ executed and called from device
- __shared__  memory space specifie
    - Resides in the shared memory space of a thread block,
    - Has the lifetime of the block,
    - Has a distinct object per block,
    - Is only accessible from all the threads within the block,

## Memory fence

The CUDA programming model assumes a device with **a weakly ordered memory system**—that is, the order in which a CUDA thread writes data to shared memory, global memory, page-locked host memory, or peer device memory is not necessarily the order in which another CUDA or host thread observes those writes. Without synchronization, it is undefined behavior for two threads to read from or write to the same memory location.

- __threadfence_block()
- __threadfence()

A common use case occurs when threads consume data produced by other threads, as shown in the following code sample—a kernel that computes the sum of an array of N numbers in a single call. Each block first sums a subset of the array and stores its result in global memory. After all blocks have finished, the last block reads each partial sum from global memory and adds them to obtain the final result. To determine which block finishes last, each block atomically increments a counter to signal it has completed computing and storing its partial sum (see Atomic Functions). The last block is the one that receives the counter value equal to `gridDim.x - 1`. If no memory fence is placed between storing the partial sum and incrementing the counter, the counter might be incremented before the partial sum is actually stored. As a result, it could reach `gridDim.x - 1` and allow the last block to begin reading partial sums before they have been updated in memory.

Memory fence functions only affect the order of a thread’s memory operations—they do not, by themselves, guarantee that these operations are visible to other threads (unlike `__syncthreads()`). In the code sample below, visibility of memory operations on the result variable is ensured by declaring it as `volatile`.

## Synchronization functions

- __syncthreads(): Waits until all threads in the block have reached this point—ensuring all global and shared memory accesses made by these threads before __syncthreads() are now visible to every thread in the block.
- A memory fence ensures memory accesses are flushed to memory, but does not block thread execution as synchronization does.
- __syncwarp() causes the executing thread to wait until all warp lanes specified in the mask have executed a __syncwarp() (with the same mask) before resuming execution.

# Chapter 11. Cooperative Groups

- Fine-grained definition a group of threads.
- Fine-grained control (e.g. synchronization) over a group of threads

# Chapter 12. Cluster Launch Control

When tackling problems of varying size, there are two primary approaches to determining the number of kernel thread blocks.

1. Approach 1: Fixed Work per Thread Block:
    1. Here, the number of thread blocks is set by the problem size, while the work assigned to each block remains constant or capped.
    2. The key advantage: better load balancing across SMs.
2. Approach 2: Fixed Number of Thread Blocks:
    1. In this method—often implemented with a block-stride or grid-stride loop—the number of thread blocks is independent of the problem size.
    2. The main advantage is reduced thread block overhead. For instance, in convolution kernels, a prologue that computes convolution coefficients—independent of the thread block index—can be run fewer times, minimizing redundant work.

## Others

Time function: measures the wall clock time, rather than time spent on execution. The former number is greater than the latter since threads are time sliced

Atomic functions: the read-modify-write operation is performed on each element of the vector residing in global memory. 

# Useful specifications

- Chapter 17. Mathematical Functions: standard functions provided by CUDA, e.g. exp(), sqrt(), etc.
- Chapter 20. Compute Capabilities: Compute capability per generation, e.g. max num of resident warps per SM, num of 32-bit registers per SM
- Chapter 18. C++ Language Support: support to C++ languages
-