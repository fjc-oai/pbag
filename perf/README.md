# Table of Content
- [Table of Content](#table-of-content)
- [Nsys](#nsys)
  - [Inspect nsys profile](#inspect-nsys-profile)
  - [Annotation](#annotation)
- [CUDA Threading Model](#cuda-threading-model)
- [Bandwidth Test](#bandwidth-test)
  - [HBM to SMEM](#hbm-to-smem)
    - [Disk \<\> Host Mem](#disk--host-mem)
- [Triton](#triton)
- [Numbers](#numbers)



# Nsys

## Inspect nsys profile
- To find a particular kernel, e.g. add2_bwd, 
  1. select cuda all streams, 
  2. click show in event view , 
  3. search for add2_bwd
  4. (can search either a kernel or block annotation)
- Nsys offers pretty handy event view in nested tree format!
- For a kernel, grid info could be useful


## Annotation
- nvtx.range_push() inserts a special event to cuda stream to mark the beginning of a range. nvtx.range_pop() inserts another special event to cuda stream to mark the end of a range. 

  - In nsys cuda stream view, the start time is the first kernel start time. The end time is the last kernel end time. Any python side duration (e.g. time.sleep()) is omitted. 

  - For multiple streams, nvtx.range_push() and nvtx.range_pop() insert events to each stream independently. The duration and annotation in nsys is also generated independently for each stream.

  - Checkout `nvtx.py`

- If using triton, kernel names are the exact triton function name


# CUDA Threading Model

- A block is a group of threads, a grip is a group of blocks. Triton abstracts away threads whereas developers can focus on block level programming. A small num of block in profile usually is not a good sign.
- Need more investigation
    - [ ] Threads within a block share some memory. What does it mean?
    - [ ] What're the properties for thread/block within a grid?
    - [ ] How are those related to warp and SM? 
    - [ ] How does the num of blocks affect computation parallelism and also memory bandwidth?
    - [ ] How does Triton translate a block level function into thread level logics?
    - [ ] etc
  

# Bandwidth Test

## HBM to SMEM
- Spec value: A100 is 1.6TB/s, H100 is 3.0TB/s
- Test value: A100 ~1TB, H100 ~1.3TB/s
  - `python perf/bandwidth_test/hbm_2_smem.py --auto`
- Need more investigation
  - [ ] How to correctly reducing the impact of cache in test bandwidth 
  - [ ] How does numk of kernel block/grid affecting mem bandwidth? More parallelism, or more contention?
  
### Disk <> Host Mem
- Mem write to disk: ~500MB/s
- Mem read from disk: 2~5GB/s


# Triton
N00b 101s
- Thread vs program instance
  - Thread: smallest execution unit
  - Program instance: thread block
  - grid -> block -> thread
- How to specify the num of threads to include in a program instance in Triton?
  - You don't. Triton abstract away the concept of threads, but programs at block/program instance level
- `tl.debug_barrier()`
  - does it synchronize all the threads within a program instance, or synchronize all  program instances?
    - Threads within a program instance are already implicitly synchronized, since SPMD execution
    - So it sounds like this synchronize all the program instances
    - However, triton doc says: Insert a barrier to synchronize all threads in a block.
???
- `tl.cumsum()`
  - Can be magical!
  - e.g. compute starting offset for each chunk of data, thus parallel the computation and store 
  - e.g. when combined with `tl.where()`
    - a list of elements, some of which satisfy some requirements while some do not
    - the goal is to reorder the list so that satisfying elements are placed at the beginning while others at the end
    - `locations = tl.cumsum(satisfied)`
    - `locations = tl.where(satisfied, locations, reversed_locations)`
- `tl.load()` & `tl.store()`
  - Pointers don't have to be consecutive. 
  - Precomputing a tensor as pointers can achieve purposed reordering


# Numbers
- On H100
  - Dense matmul
    - [4k, 700] @ [700, 100k] ~1ms
    - [4k, 700] @ [700, 400k] ~4ms
  
# Profile Python Code
## Profilers
1. Sampling based profiler: get stack traces at fixed time intervals, e.g.
    - `Scalene`: register signal handler for target python process, and record stack traces when signal is triggered, by C++
    - `py-spy`: fetches stack trace of a remote python process by virtual memory address, by rust
2. Trace based profiler: get stack traces and frame info for each function call
    - `yappi`: register hooks through python C API PyEval_SetProfile(), record frame and stack trace for each function call

## Overhead
- `Scalene` and `py-spy`: <10% based on reported results
- python trace based profiler: ~15x overhead
- c++ trace based profiler: ~10x overhead
```
cd perf/profiler
python setup.py build_ext --inplace
python benchmark_profiler.py

Baseline: 1.03 seconds
Python profiler: 16.00 seconds
{'duration:38': 1, 'task:33': 3, 'fibonacci:27': 65673000}
C++ profiler: 11.03 seconds
{'27:fibonacci': 65673000, '33:task': 3, '38:duration': 1}
```

## Timers
- `time.perf_counter()`: wall time
- `time.thread_time()`: cpu time
- `torch.cuda.synchronize()`: the behavior seems similar to spin lock. It blocks the thread, check ready, yields the GIL to other threads before checking again.