"""
* Larger matrix usually results into better GPU utilization and compute throughput.

* Once exceeding a certain threshold, further increasing matrix size results in diminishing returns.

* If we run small matrix multiplication for a larger number of times, will we get equivalent performance?
  * Without cudagraph, kernel launch overhead is significant. Observed significant throughput degradation.
  * With cudagraph, the throughput is much better but still not as good as large matrix multiplication. 
    This is likely due to 1) cuda kernel won't be able to fully utilize GPU resources when the matrix size 
    is small, 2) each kernel launch/execution still has some fixed non-trivial overhead.

(m, n)                                   Cudagraph          min        max        avg
---------------------------------------  -----------  ---------  ---------  ---------
matmul((4096,4096), (4096,4096)) * 16    N             0.116039   0.116039   0.116039
matmul((4096,4096), (4096,4096)) * 16    Y             0.115843   0.115843   0.115843
matmul((2048,2048), (2048,2048)) * 128   N             0.124656   0.124656   0.124656
matmul((2048,2048), (2048,2048)) * 128   Y             0.124366   0.124366   0.124366
matmul((1024,1024), (1024,1024)) * 1024  N             0.132008   0.132008   0.132008
matmul((1024,1024), (1024,1024)) * 1024  Y             0.132936   0.132936   0.132936
matmul((512,512), (512,512)) * 8192      N             0.781121   0.781121   0.781121
matmul((512,512), (512,512)) * 8192      Y             0.211903   0.211903   0.211903
matmul((256,256), (256,256)) * 65536     N             6.37002    6.37002    6.37002
matmul((256,256), (256,256)) * 65536     Y             0.617728   0.617728   0.617728
matmul((128,128), (128,128)) * 524288    N            49.7778    49.7778    49.7778
matmul((128,128), (128,128)) * 524288    Y             3.22455    3.22455    3.22455

"""
import time
from collections import defaultdict

import torch
from tabulate import tabulate


def test(m, n):
    a = torch.randn((m, m), device='cuda')
    b = torch.randn((m, m), device='cuda')
    torch.cuda.synchronize()
    st = time.time()
    for _ in range(n):
        torch.mm(a, b)
    torch.cuda.synchronize()
    return time.time() - st

def test_with_cudagraph(m, n):
    a = torch.randn((m, m), device='cuda')
    b = torch.randn((m, m), device='cuda')
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            torch.mm(a, b)
    torch.cuda.synchronize()
    st = time.time()
    g.replay()
    torch.cuda.synchronize()
    return time.time() - st

def benchmark():
    M = 1024 * 8
    N = 2
    N_DECAY = 6
    N_ITER = 2
    N_WARMUP = 1
    res = defaultdict(list)
    for use_cudagraph in [False, True]:
        print(f'Use cudagraph: {use_cudagraph}')
        for i in range(N_ITER):
            print(f'Iteration {i+1}/{N_ITER}')
            for j in range(N_DECAY):
                print(f'  Decay {j+1}/{N_DECAY}')
                scale = 2 ** (j+1)
                m = M // scale
                n = N * (scale ** 3)
                if use_cudagraph:
                    time = test_with_cudagraph(m, n)
                else:
                    time = test(m, n)
                if i >= N_WARMUP:
                    res[(m, n)].append((time, use_cudagraph))
    table = []
    for (m, n), v in res.items():
        no_cudagraph = [x[0] for x in v if not x[1]]
        cudagraph = [x[0] for x in v if x[1]]
        table.append((f"matmul(({m},{m}), ({m},{m})) * {n}", 'N', min(no_cudagraph), max(no_cudagraph), sum(no_cudagraph) / len(no_cudagraph)))
        table.append((f"matmul(({m},{m}), ({m},{m})) * {n}", 'Y', min(cudagraph), max(cudagraph), sum(cudagraph) / len(cudagraph)))
    print(tabulate(table, headers=['(m, n)', 'Cudagraph', 'min', 'max', 'avg']))

benchmark()


    
