# Table of Content
- [Table of Content](#table-of-content)
- [Nsys](#nsys)
  - [Inspect nsys profile](#inspect-nsys-profile)
  - [Annotation](#annotation)
- [CUDA Threading Model](#cuda-threading-model)
- [Bandwidth Test](#bandwidth-test)
  - [HBM to SMEM](#hbm-to-smem)



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
  
