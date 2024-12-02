# Table of Content
- [Table of Content](#table-of-content)
- [Memory Management](#memory-management)
  - [Links](#links)
  - [Cuda](#cuda)
  - [PyTorch CudaCacheAllocator (CCA)](#pytorch-cudacacheallocator-cca)
  - [Monitoring](#monitoring)

# Memory Management

## Links
- Gentle intro to memory management. A great one!
    - https://dev-discuss.pytorch.org/t/fsdp-cudacachingallocator-an-outsider-newb-perspective/1486
- PyTorch cuda cache allocator internal details
    - https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html 
- PyTorch CUDA mem usage
    - https://pytorch.org/docs/stable/torch_cuda_memory.html

## Cuda 
(so-heard knowledges)
1. `cudaMalloc()` and `cudaFree()` are host-side functions. Synchronous functions. Don't issue kernels that run on Cuda streams.
    - It imples Cuda memory is managed by runtime library running on Cpu side. Device executes computation upon whatever addresses it receives

2. `cudaMalloc()` is not that slow, just directly claims a unused memory.
3. `cudaFree()` triggers a device synchronization, waiting for all cuda ops completion across streams.

## PyTorch CudaCacheAllocator (CCA)
1. Single stream case is simple. All of the tensor allocation and deallocation happen on Cpu without any synchronization needed. Gpu operations runs on Cuda queue following the exactly same order. 

2. CCA creates one-pool-per-stream. When no cross stream tensor access, the case is as simple as single stream.

3. Cross stream tensor access without proper synchronization yields to race condition. See example below.

4. `tensor.record_stream()`: tensor's deallocator won't immediately free the underlying memory. It instead saves the event info of recorded stream into tensor's (or some CCA's) metadata. During a later `malloc`, CCA will evaluate if this block of memory is reusable based on event's status. 

5. Besides `tensor.record_stream()`, introducing stream level synchronization is another way to avoid race condition. 

6. During malloc if there is not enough contiguous free memory block in CCA to allocate, (might due to fragementation) it triggers a `cudaFree()`. It returns all the unused memory back to Cuda. Cuda is able to reshuffle those noncontiguous memory to contiguous blocks by virtual memory remapping! It 
    - When this happens, it could significantly hurt the perf due to device synchronization.
    - If not enough memory available even after `cudaFree()`, OOM exception will be raised
    - Takeaway: eagerly allocate major memory ahead of time if possible, e.g. params and grads in a layer, instead of a frequently allocate and free on the fly.

```
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["duration"])
def _sleep(duration):
    start = tl.extra.cuda.globaltimer()
    while tl.extra.cuda.globaltimer() - start < duration:
        pass


def sleep(duration: float):
    _sleep[(1,)](int(duration * 1e9))


N = 10
AVOID_RACE_CONDITION_BY_RECORD_STREAM = False
AVOID_RACE_CONDITION_BY_STREAM_SYNC = False

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()


for i in range(N):
    with torch.cuda.stream(s1):
        x = torch.full((1024,), 1, device="cuda", dtype=torch.float32)
        # Without this line, the first iteration may succeed. From ChatGpt:
        # "Without the in-place operation, x's data might not be fully allocated
        # or utilized in the same way. The memory allocator may decide not to
        # reuse x's memory immediately because it wasn't actively used.
        x *= 2
        evt = torch.cuda.current_stream().record_event()

    with torch.cuda.stream(s2):
        sleep(1)
        torch.cuda.current_stream().wait_event(evt)
        y = x * 3

    if AVOID_RACE_CONDITION_BY_RECORD_STREAM:
        x.record_stream(s2)
    
    if AVOID_RACE_CONDITION_BY_STREAM_SYNC:
        s1.wait_stream(s2)

    # Allocating a new tensor with the same size on the same stream will reuse
    # the memory freed by x, thus triggering the race condition.
    del x
    with torch.cuda.stream(s1):
        c = torch.full((1024,), 31, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()

    expected = torch.full((1024,), 6, device="cuda", dtype=torch.float32)
    print(f"iter {i} {torch.equal(y, expected)=}")

```
## Monitoring
1. `torch.cuda.memory_allocated()` and `torch.cuda.memory_allocated()`
2. `torch.cuda.memory_allocated()` will count a memory as freed as long as tensor is deleted, without considering if it's recorded on another stream.
3. `torch.cuda.memory._record_memory_history()` to track runtime memory allocation history overtime with stacktrace. 
    - https://pytorch.org/docs/stable/torch_cuda_memory.html