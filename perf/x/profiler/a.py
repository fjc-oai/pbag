import time

import cprofiler

use_profiler = True
# use_profiler = False
if use_profiler:
    # Enable profiling: this sets our C-level trace function.
    cprofiler.set_profile()

def heavy_computation(n):
    total = 0

    # Instead of multiple small functions, we inline the arithmetic.
    def compute_value(i, j):
        prod = i * j
        sum_val = i + j
        inc_sum = sum_val + 1
        return prod % inc_sum

    # Process a batch of computed results. This function is called every ~10 operations.
    def process_batch(batch):
        result = 0
        for value in batch:
            result += value
        return result

    batch = []
    for i in range(n):
        for j in range(n):
            batch.append(compute_value(i, j))
            if len(batch) == 10:
                total += process_batch(batch)
                batch = []
    if batch:  # Process remaining operations if they don't fill a complete batch.
        total += process_batch(batch)
    return total

def foo():
    print("Starting heavy computation...")
    result = heavy_computation(1000)
    print("Computation result:", result)

start = time.perf_counter()
foo()
end = time.perf_counter()
print(f"Time taken: {end - start} seconds")

# Disable profiling.
if use_profiler:
    cprofiler.unset_profile()