# https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59
# nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --cudabacktrace=true -x true -o my_profile python nvtx.py
import time

import torch


def cuda_sleep(x):
    N_CYCLES = 1_000_000_000
    torch.cuda._sleep(x * N_CYCLES)

def profile_1():
    """
    nvtx.range_push() inserts a special event to cuda stream to mark the
    beginning of a range. nvtx.range_pop() inserts another special event to cuda
    stream to mark the end of a range. 

    In nsys cuda stream view, the start time is the first kernel start time. The
    end time is the last kernel end time. Any python side duration (e.g.
    time.sleep()) is omitted.

    In nsys python thread view, the start and end time exactly measures the
    python/cpu side duration.
    """
    torch.cuda.nvtx.range_push("l1")
    time.sleep(5)
    cuda_sleep(1)
    time.sleep(5)
    torch.cuda.nvtx.range_pop()

def profile_2():
    """ 
    For multiple streams, nvtx.range_push() and nvtx.range_pop() insert events
    to each stream independently. The duration and annotation in nsys is also
    generated independently for each stream.

    e.g. event "l1" is rendered in both stream 2 and stream 3. The duration is 
    5s and 2s respectively. 
    """
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    s3 = torch.cuda.Stream()
    for idx, s in enumerate([s1, s2, s3]):
        print(f"stream {idx}: {s.stream_id}")

    with torch.cuda.stream(s1):
        cuda_sleep(10)
    
    time.sleep(1)
    torch.cuda.nvtx.range_push("l1")
    with torch.cuda.stream(s2):
        cuda_sleep(5)
    
    time.sleep(1)
    with torch.cuda.stream(s3):
        cuda_sleep(2)

    time.sleep(1)
    torch.cuda.nvtx.range_pop()    

def profile_3():
    """ 
    Nested nvtx.range_push() and nvtx.range_pop() 
    """
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        torch.cuda.nvtx.range_push("all_three")
        cuda_sleep(1)
        torch.cuda.nvtx.range_push("last_two")
        cuda_sleep(1)
        torch.cuda.nvtx.range_push("last_one")
        cuda_sleep(1)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()

def profile_4():
    """
    Seems it doesn't allow to push and pop nvtx range for a particular stream.
    It automatically applies to all streams.
    """
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    with torch.cuda.stream(s1):
        torch.cuda.nvtx.range_push("s1")
        cuda_sleep(1)
    
    with torch.cuda.stream(s2):
        cuda_sleep(1)
    torch.cuda.nvtx.range_pop()

def main():
    torch.cuda.cudart().cudaProfilerStart()
    profile_4()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
