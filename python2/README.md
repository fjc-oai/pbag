- [Thread](#thread)
  - [Context switch](#context-switch)
  - [Thread creation overhead](#thread-creation-overhead)
- [Asyncio](#asyncio)
  - [Cooperative multitask under the hood](#cooperative-multitask-under-the-hood)
  - [Debug hanging issues](#debug-hanging-issues)
  - [Exception handling](#exception-handling)
  - [Thread in Asyncio](#thread-in-asyncio)
- [Context Var](#context-var)
- [Itertools](#itertools)
- [Memory Management](#memory-management)
  - [Questions to answer](#questions-to-answer)
  - [Major concepts](#major-concepts)
  - [Execution flow](#execution-flow)
  - [mmap](#mmap)
  - [Pytorch tensor](#pytorch-tensor)
  - [Profiling tools](#profiling-tools)

# Thread

##  Context switch 

1. What is a python thread? Is it a pure application level object?
    - No. It uses OS thread under the hood
    - A normal OS thread can start execution as long as its state becomes to ready-to-run
    - A python thread essentially executes python byte-code. In order to exectute, it first requires underlying OS thread is ready-to-run, and then requires to acquare GIL.

2. How does python thread scheduling & context switch work? 
    - Voluntary (yield) and non-voluntary context switch (preemption)
    - Yield: sleep(), io.wait(), lock.acquire(), etc
    - Preemption: interpreter forces a context switch on a thread after a certain number of byte-code executed, which is controlled by `sys.setswitchinterval()`.

3. Why excessive theads can slow down the program?
    - More threads compete for GIL
    - More threads to schedule on Otertools)


4. How to benchmark context switch overhead?
    -  `python context_switch_overhead.py`
    - Complex arithmatic operations (hard to opt away), memory access, etc.
    - Otherwise won't be easy to repro with loop over simple operations (e.g. float mul only)

5. Why profiler doesn't help in this case?
    - Profiler, e.g. py-spy doesn't show context switch overhead
    - It has no visibility into system-level scheduling, and only samples python-level threads, but not OS level.


## Thread creation overhead

- Benchmark: `python thread_creation_overhead.py`

- Always create a new thread on-the-fly vs using a global thread pool executor

- Intuitively, thread creation itself is expensive. So it'll be slower (plus thread context switch overhead).

- From benchmark, the difference is not that significant though, ~10%.


# Asyncio

## Cooperative multitask under the hood

- https://github.com/fmars/tiny-asyncio

- https://docs.python.org/3/library/asyncio.html


## Debug hanging issues

- It's not uncommon to see silent hanging issues in asyncio. Two common cases are

1. Running Cpu work on eventloop, which is either expensive/slow or block, thus blocked the eventloop

2. Silent exception: similar to threads, exception raised in async task won't be automatically propagated and thus silently swallowed, especailly the task is running on worker task. 

    - Remember to explicitly catch the exception, and propagate by `future.set_result(ex)`

## Exception handling

- asyncio.wait_for() throws TimeoutException

- Eventloop shutdown (e.g. from destructor) will cancel all the pending tasks, e.g. pending `queue.get()`!



## Thread in Asyncio 

- Demo: `python thread_in_asyncio.py`

- Works in a almost the same way as io/select implementation

- `run_in_executor()` runs the enclosed task in the executor on  one hand, it wraps the returned `concurrent.Future` with `asyncio.Future` which monitored by eventloop on the other hand. 

- Once the underlyting `concurrent.Future` is completed, the corresponding `asyncio.Future` will be marked done as well, thus be ready to be picked up by eventloop to schedule remaining code in its upstream task.

- If underlying function runs asynchronously and returns a concurrent.futures.Future, extra attention is required!
    1. asyncio.wrap_future() to wrap the returned concurrent.futures.Future and await it
    2. try-except the block, and propagate the execution properly

# Context Var

1. Basics
   - Manage context-local state, used mainly in async programming (e.g. asyncio) 
   - Each context (e.g. coro/async task) has its own value for a variable
   - Similar to request context commonly used in C++ backend services
2. Context var vs thread-local
   - Context var: mainly used in asyncio. All the tasks run on the same threads
   - Thread-local: mainly used in multi-threading. Each task runs on its own thread


# Itertools

1. `it1 = itertools.count(start=1)`

2. `it2 = more_itertools.batched(it1, 5)`


# Memory Management

## Questions to answer
1. How does Python allocate memory? 
2. What needs to be intercepted to profile memory allocation?
3. How are `pymalloc`, `malloc`, `sbrk`, `mmap` related?

## Major concepts
1. The allocator is responsible for, 1) reserving memory from OS, 2) allocating slices of memory to objects
   1. Reservation: the allocator uses `mmap` or `sbrk` to request memory from OS
   2. Allocation: allocator maintains a data structure (e.g. block tree) to efficiently allocate memory chunk, optimizing for low latency and reduced fragmentation.
2. Virtual vs physical memory
   1. Allocator deals with virtual memory, both for reservation and allocation. It has no visibility into phisical memory. 
   2. A page fault is triggered when accessing an virtual memoery, handled by OS, to map a corresponding phisical memory page.

## Execution flow
1. When a process starts, OS assigns a unique virtual memory space divided into disjoint sections, e.g. code segment, stack segment, heap segment, etc
2. When application code creates a new object, it triggers a memory allocation call.
3. `PyMalloc` handles the allocation request first:
   1. If it's a python special object (e.g. integer), PyMalloc optimizes the allocation.
   2. Otherwise it forwards the request to `glibc`'s `malloc`.
   3. Note that, memory allocation from native code (e.g. C) bypasses `PyMalloc` and directly uses `malloc`
4. malloc() allocates memory from heap, if there is sufficient available memory, using some algorithm, optimizing for low latency, low fragementation, etc.
   1. If sufficient heap memory is unavailable, depending on the request size, 
   2. small requests: `malloc` invoeks kernels's `sbrk` to expand the heap's contiguous memory
   3. large requests (e.g. >128KB?): `malloc` uses `mmap` to reserve a separate large memory region\

## mmap
1. mmap serves for several difference purposes
2. In malloc case, it creates anonemous map, which basically reserves a large virtual memory region from OS. Don't realy understand why it's still called map in this case, likely some legacy reason?
   1. When a previously malloc calling mmap allocated memory is freed, the allocator calls munmap to free it
   2. This behavior might be further optimized by letting allocator to hold the region for reuse instead of instead of inmediate return, to save kernel invocations. jemalloc does this?.
3. It can also map a file to virtual memory space, so to save an extra memory copy to access the file.

## Pytorch tensor
1. GPU tensor allocation increases both host virtual memory (virt) and resident memory (res).
2. The GPU context reserves a virtual memory region.
3. Bookkeeping data structures contribute to resident memory usage.


## Profiling tools
1. top cmd output: 
   1. `virt`: virtual memory - allocated logical memory 
   2. `res`: residual memory - physical memory
   3. `shr`: shared memory, e.g. glibc, /dev/shm, etc
2. Tracemalloc:
   1. Likely hooks into PyMalloc  
   2. Tracks only allocated but not yet freed memory, useful for debugging memory leaks.
   3. For PyTorch tensors, it tracks Python-allocated metadata but not the native-allocated data (e.g., tensor data).
3. Memray:
    1. Likely hooks into malloc, enabling tracing of both Python and native memory allocations.   
    2. https://github.com/bloomberg/memray/blob/ef3d3ea2e696de5d01bb94879abf27159b989375/src/memray/_memray/hooks.cpp#L116 


